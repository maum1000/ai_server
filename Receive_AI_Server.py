import asyncio
import struct
import json
import cv2

import ai_system.ai_pybo



REQUEST_IMAGE_1 = 100
REQUEST_IMAGE_2 = 101
REQUEST_AI_ANALYSIS = 200
RESPONSE_AI_ANALYSIS = 210
READY_AI_ANALYSIS  = 220

request_ai_ready = "REQ_AI_READY"
request_ai_ready_ok = 'REQ_AI_READY_OK'
request_ai_analysis_image = 'REQ_AI_IMAGE'
request_ai_analysis_image_ok = 'REQ_AI_IMAGE_OK'

class ClientConnectionHandler:

    def __init__(self,reader, writer):

        self.reader = reader
        self.writer = writer

        self.addr = writer.get_extra_info('peername')
        self.bFirstPacket = False

    async def handle(self):

        try:
            await self.read_loop()
        except Exception as e:
            print(f"{self.addr}에서 처리되지 않은 에러 발생 {e}")


    async def protocol_send_data(self, protocol, data):
        #protocol을 붙이고 뒤에 데이타 보내기

        data_protocol = struct.pack('>I', protocol)  # Big-endian 형식으로 변환
        real_data = struct.pack('I',data)
        data_protocol+=real_data
        self.writer.write(data_protocol)
        await self.writer.drain()
        print("protocol send data ", protocol)

    async def protocol_send_data_json(self, protocol, data, isImage= False):

            if isImage == False :
                response = {"protocol": protocol, "message": data}
            else:
                response = {"protocol": protocol,"image_size":len(data)}

            try:
                self.writer.write((json.dumps(response)+'\n').encode())
                await self.writer.drain()
            except Exception as e:
                print("Exception Send Data Json",e)

            if isImage ==True:
                self.writer.write(data)
                await self.writer.drain()
                print("send to client analyimg")




    async def read_process(self, packet):

            #packet read
            # print("패킷 사이즈",len(packet))
            # protocol_id = int.from_bytes(packet[:4], 'big') #big endian
            # print("read_process : ", protocol_id)
            # if protocol_id == REQUEST_AI_ANALYSIS :
            #     await self.protocol_send_data(READY_AI_ANALYSIS,300)
            #json 데이타
            message = json.loads(packet.decode())
            print("received json  message", message)


            protocol = message['protocol']

            if protocol==request_ai_ready: #첫번째 요청
                await self.protocol_send_data_json(request_ai_ready_ok,"")
            elif protocol==request_ai_analysis_image:  #이미지 분석 요청
                print("request _ai bFirstPacket true")
                self.bFirstPacket = True
                # image_data = base64_decode(message['image'])
                # with open("received_image2222.png", "wb") as f:
                #     f.write(image_data)
                # await self.protocol_send_data_json(request_ai_analysis_image_ok,"")



    async def read_loop(self):
        read_data = b""
        total_bytes = 0
        print("read loop=========================")
        while True:
            try:
                    packet = await self.reader.readline()

                    if not packet:
                         break
                    # read_data+=packet

                    message = json.loads(packet.decode())
                    print("received json  message", message)

                    protocol = message['protocol']

                    if protocol == request_ai_ready:  # 첫번째 요청
                        await self.protocol_send_data_json(request_ai_ready_ok, "")
                    elif protocol == request_ai_analysis_image:  # 이미지 분석 요청
                        #이미지 사이즈를 받아야함
                        img_size = message['image_size']

                        print("image size = ",img_size)
                        image_data = b""
                        while len(image_data) <img_size :
                            chunk = await self.reader.read(img_size - len(image_data))
                            if not chunk:
                                break
                            image_data +=chunk

                        if image_data:
                           img_new = ai_system.ai_pybo.start_ai_check_image_buffer(image_data)
                           #cv2.imwrite("test_result.jpg", cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR))
                           #result , buffer = cv2.imencode(".jpg",cv2.cvtColor(img_new,cv2.COLOR_RGB2BGR))
                           result, buffer = cv2.imencode(".jpg", img_new)
                           if not result:
                               print("result error===== ")
                           send_buffer = buffer.tobytes()
                           await self.protocol_send_data_json(request_ai_analysis_image_ok,send_buffer,True)
                           # 이미지를 보내기

            except ConnectionResetError:
                       print(f"{self.addr}와의 연결이 끊어졌습니다.")

            except Exception as e:
                       print("error e",e)


        print("이미지22222222222222222222222222",total_bytes)
        #if(self.bFirstPacket==True):
        #    print('bfirstpacket = True')
            # image_data = base64_decode(read_data['image'])
            # with open("received_image2222.png", "wb") as f:
            #     f.write(image_data)
            # await self.protocol_send_data_json(request_ai_analysis_image_ok,"")

        self.writer.close()
        await self.writer.wait_closed()
        #AI작업으로 처리해야함

        # if read_data:
        #     with open('received_image.jpg', 'wb') as f:
        #         f.write(read_data)
        #         print("이미지 잘 받았음")

        # image buffer로 처리하기
        #ai_process = AIImage_Process(image_path_ori)
        #output_path = ai_process.process_image_similarity(image_path_ori, image_path_dest)
        #return output_path



async  def client_connected_callback(reader, writer):

            handler = ClientConnectionHandler(reader,writer)
            await handler.handle()



class Server:

    def __init__(self,host:str='127.0.0.1',port:int=7777):
        self.host = host
        self.port = port


    async def run(self):

        server = await asyncio.start_server(client_connected_callback, host=self.host, port = self.port)
        addrs=','.join(str(sock.getsockname())for sock in server.sockets)
        print(f"서버 실행중 : {addrs}")

        async with server:
            await server.serve_forever()



if __name__=='__main__':

    server= Server()
    asyncio.run(server.run())
