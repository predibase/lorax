import grpc
from lorax_server.pb import generate_pb2, generate_pb2_grpc


def run(
    input_path: str,
    output_path: str,
    uds_path: str = "/tmp/lorax-server",
):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = generate_pb2_grpc.LoraxServiceStub(channel)
        resp = stub.Health(generate_pb2.HealthRequest())
        print("HEALTH RESPONSE", resp)
        # response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
        # print("Greeter client received: " + response.message)
        # response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='you'))
        # print("Greeter client received: " + response.message)
