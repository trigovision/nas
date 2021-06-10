from pose.train_dist import parse_args, train_dist

if __name__ == "__main__":
    args = parse_args()
    train_dist(args, "nas/pose/train/train.py")
