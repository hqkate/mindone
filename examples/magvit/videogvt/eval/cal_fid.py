import os
from fid.fid import FrechetInceptionDistance


def calculate_fid(real_imgs, generated_imgs):
    fid_scorer = FrechetInceptionDistance(batch_size=len(real_imgs))
    fid_score = fid_scorer.compute(generated_imgs, real_imgs)
    return fid_score


if __name__ == "__main__":
    gen_imgs = ["data/img_1.jpg", "data/img_2.jpg"]
    gt_imgs = [
        "data/img_10.jpg",
        "data/img_11.jpg",
    ]

    # fid_scorer = FrechetInceptionDistance("./inception_v3_fid.ckpt")
    fid_scorer = FrechetInceptionDistance()
    score = fid_scorer.compute(gen_imgs, gt_imgs)
    print("ms FID: ", score)