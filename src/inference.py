def generate_labels(preds, id_to_label):
    labels = []
    for k, v in id_to_label.items():
        if preds[k] == 1:
            labels.append(v)
    return labels
	
def inference():
    for _, row in test_df.iterrows():
        # print (row)
        img_id = row["Sample_ID"]
        img_path = f"{config.data_path}/{img_id}/image.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_transform(image=img)["image"]
        img = img / 255
        _, h, w = img.shape
        square = torch.zeros((3, 256, 256))
        square[:, 0:h, 0:w] = img
        square = square.unsqueeze(0)
        square = square.to(config.device)
        preds = net(square)[0]
        preds = preds.data.cpu().numpy()
        preds = np.where(preds > 0.5, 1.0, 0.0)
        preds = generate_labels(preds, pet_id_to_labels)
        print(row["Breed"], "|", preds)