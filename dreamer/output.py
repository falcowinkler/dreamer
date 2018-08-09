from data.dataset import download_file
import os
import compiled_protobuf.tile_data_pb2 as tile_data_pb2
import numpy as np

def compile_protobuf():
    download_file("https://raw.githubusercontent.com/falcowinkler/copicat/master/resources/proto/tile_data.proto",
                  "../proto/tile_data.proto")

    os.system("protoc --python_out ../compiled_protobuf -I ../proto ../proto/*.proto")


def to_protobuf(generator_output, iteration):

    compile_protobuf()
    for i, generated_level in enumerate(generator_output[0]):
        tile_data = tile_data_pb2.TileData()
        tile_data.raw_data = bytes(np.reshape(generated_level, [23 * 31]).tolist())
        f = open("../out/generated_level_it_" + str(iteration) + "_sample_number_" + str(i) + ".pb", "wb")
        f.write(tile_data.SerializeToString())
        f.close()
