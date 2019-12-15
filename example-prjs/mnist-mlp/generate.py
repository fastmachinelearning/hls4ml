import argparse

def main(args):
    image_shape = str(args.image_size) + 'x' + str(args.image_size)
    pixel_count = args.image_size * args.image_size

    input_file = 'templates_yml/keras-config-template.yml'
    output_file = 'keras-config-' + image_shape + '-fxd-' + str(args.word_size) + '-' + str(args.integer_size) + '.yml'
    data_type = 'ap_fixed<' + str(args.word_size) + ',' + str(args.integer_size) + '>'

    fin = open(input_file, 'rt')
    fout = open(output_file, 'wt')

    for line in fin:
        line = line.replace('>>>DATA_TYPE<<<', data_type)
        line = line.replace('>>>PIXEL_COUNT<<<', str(pixel_count))
        line = line.replace('>>>IMAGE_SHAPE<<<', image_shape)
        line = line.replace('>>>WORD_SIZE<<<', str(args.word_size))
        line = line.replace('>>>INTEGER_SIZE<<<', str(args.integer_size))
        fout.write(line)

    fin.close()
    fout.close()

if __name__ == '__main__':
    word_size_default = 18
    integer_size_default = 8
    image_size_default = '28'

    parser = argparse.ArgumentParser(description='hls4ml generator')
    parser.add_argument('--word_size', help='Fixed-point word size [' + str(word_size_default) + ']', action='store', dest='word_size', type=int, default=word_size_default)
    parser.add_argument('--integer_size', help='Fixed-point integer-part size [' + str(integer_size_default) + ']', action='store', dest='integer_size', type=int, default=integer_size_default)
    parser.add_argument('--image_size', help='Model [' + str(image_size_default) + ']', action='store', dest='image_size', type=int, choices=[12,14,16,28], default=image_size_default)
    args = parser.parse_args()
    main(args)

