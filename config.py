import argparse

def get_args():
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--size', '-s', type=int, default=299, 
                        help='The size of the image')
    parser.add_argument('--path', '-p', type=str,
                        help='The path to the data')
    parser.add_argument('--m_label', type=bool, default=True,
                        help='Multi-labelling classification')
    parser.add_argument('--format', '-f', type=str, default='.jpeg',
                        help='The type of format for the data')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='The test size for validation set')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='The batch size for training data')
    parser.add_argument('--acts', '-a', type=str, default='sigmoid',
                        help='The activation function for training')
    parser.add_argument('--weight', '-w', type=str,
                        help='Pretrained weights for training')
    
    args = parser.parse_args()
    
    return args