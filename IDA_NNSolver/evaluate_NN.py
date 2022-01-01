from pathlib import Path

def eval_NN():
    path = str(Path(__file__).resolve().parent) + "/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inputs
    cube_data = np.loadtxt(path + "NN_input_test.csv", delimiter=',')
    cube_data = torch.from_numpy(cube_data)

    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
    colors_in_face = torch.stack(colors_in_face)

    # Targets
    targets = np.loadtxt(path + "NN_target_test.csv", delimiter=',')
    targets = torch.tensor(targets).to(device)

    # Neural Network
    model = torch.load(path + "3x3HeuristicModel.pt").to(device)
    model.eval()

    # Predict
    pred_one_hot = model(inputs)
    pred = torch.argmax(results_one_hot)

    # Plot predictions against real values
    plt.style.use('seaborn')
    plt.plot(targets.numpy(), pred.numpy(), 'o')
    plt.plot([0, 20], [0, 20])
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.show()
