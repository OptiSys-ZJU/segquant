from dataset.caption_control_dataset import CaptionControlDataset

COCODataset = CaptionControlDataset

if __name__ == "__main__":
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/COCO-Caption2017-canny", cache_size=16
    )
    data_loader = dataset.get_dataloader(batch_size=1)

    for i, batch in enumerate(data_loader):
        prompt, image, control = batch[0]
        print("prompt:", prompt)
        print("image:", image)
        print("control:", control)
        print("--------------")
        if i > 10:
            break
