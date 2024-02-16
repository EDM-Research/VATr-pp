import argparse
from generate import generate_text, generate_authors, generate_fid, generate_page, generate_ocr, generate_ocr_msgpack
from generate.ocr import generate_ocr_reference
from util.misc import add_vatr_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=['text', 'fid', 'page', 'authors', 'ocr'])

    parser.add_argument("-s", "--style-folder", default='files/style_samples/00', type=str)
    parser.add_argument("-t", "--text", default='That\'s one small step for man, one giant leap for mankind ΑαΒβΓγΔδ', type=str)
    parser.add_argument("--text-path", default=None, type=str, help='Path to text file with texts to generate')
    parser.add_argument("-c", "--checkpoint", default='files/vatr.pth', type=str)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("--count", default=1000, type=int)
    parser.add_argument("-a", "--align", action='store_true')
    parser.add_argument("--at-once", action='store_true')
    parser.add_argument("--output-style", action='store_true')
    parser.add_argument("-d", "--dataset-path", type=str)
    parser.add_argument("--target-dataset-path", type=str, default=None)
    parser.add_argument("--charset-file", type=str, default=None)
    parser.add_argument("--interp-styles", action='store_true')

    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--fake-only", action='store_true')
    parser.add_argument("--all-epochs", action='store_true')
    parser.add_argument("--long-tail", action='store_true')
    parser.add_argument("--msgpack", action='store_true')
    parser.add_argument("--reference", action='store_true')
    parser.add_argument("--test-set", action='store_true')

    parser = add_vatr_args(parser)
    args = parser.parse_args()
    
    if args.action == 'text':
        generate_text(args)
    elif args.action == 'authors':
        generate_authors(args)
    elif args.action == 'fid':
        generate_fid(args)
    elif args.action == 'page':
        generate_page(args)
    elif args.action == 'ocr':
        if args.msgpack:
            generate_ocr_msgpack(args)
        elif args.reference:
            generate_ocr_reference(args)
        else:
            generate_ocr(args)
