import argparse
import pickle

from dataset import *
from models.cnn_block_frame_flow import CNNBlockFrameFlow
from torch.autograd import Variable

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running", 
    "walking"
]


def getscore(current_block_frame, current_block_flow_x, current_block_flow_y, mean, use_cuda = 'False'):
    block_frame = np.array(
        current_block_frame,
        dtype=np.float32).reshape((1, 15, 60, 80))

    block_flow_x = np.array(
        current_block_flow_x,
        dtype=np.float32).reshape((1, 14, 30, 40))

    block_flow_y = np.array(
        current_block_flow_y,
        dtype=np.float32).reshape((1, 14, 30, 40))

    block_frame -= mean["frames"]
    block_flow_x -= mean["flow_x"]
    block_flow_y -= mean["flow_y"]

    tensor_frames = torch.from_numpy(block_frame)
    tensor_flow_x = torch.from_numpy(block_flow_x)
    tensor_flow_y = torch.from_numpy(block_flow_y)
    
    instance_frames = Variable(tensor_frames.unsqueeze(0))
    instance_flow_x = Variable(tensor_flow_x.unsqueeze(0))
    instance_flow_y = Variable(tensor_flow_y.unsqueeze(0))

    if use_cuda == 'True':
        instance_frames = instance_frames.cuda()
        instance_flow_x = instance_flow_x.cuda()
        instance_flow_y = instance_flow_y.cuda()                  

    score = model(instance_frames, instance_flow_x,
                    instance_flow_y).data[0].cpu().numpy()
    return score


# Numbe of frames in a block is 15. A snippet consists of several blocks, generating one action recognition predicition.
# snippet_duration = 15* num_of_blocks_in_a_snippet/fps_of_input_video 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="data",
        help="directory to dataset")
    parser.add_argument("--model_dir", type=str, default="data/model_epoch46.chkpt",
        help="directory to model")
    parser.add_argument("--evaluate_which", type=str, default="dev",
        help="evaluate on which set?")   
    parser.add_argument("--meanimage_which", type=str, default="dev",
        help="use which dataset to calculate the mean image?")     
    parser.add_argument("--block_num", type=str, default="5",
        help="A snippet includes how many blocks?")     
    parser.add_argument("--use_cuda", type=str, default="True",
        help="Whether use cuda?")                   

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    evaluate_which = args.evaluate_which
    meanimage_which = args.meanimage_which
    use_cuda = args.use_cuda
    block_num = args.block_num

    print("Loading dataset")
    mean_dataset = BlockFrameFlowDataset(dataset_dir, meanimage_which)  # mean images come from which dataset
    video_frames = pickle.load(open("data/" + evaluate_which + ".p", "rb"))
    video_flows = pickle.load(open("data/" + evaluate_which+ "_flow.p", "rb"))
    mean = {}
    mean["frames"] = mean_dataset.mean["frames"]
    mean["flow_x"] = mean_dataset.mean["flow_x"]
    mean["flow_y"] = mean_dataset.mean["flow_y"] 

    print("Loading model")
    chkpt = torch.load(model_dir, map_location=lambda storage, loc: storage)
    model = CNNBlockFrameFlow()
    model.load_state_dict(chkpt["model"])
    if use_cuda == 'True':
        model.cuda()

    # Number of correct classified videos.
    correct = 0

    snippet_stat = np.zeros((6,6))
    print("%19s" %("snippet_predection"), " video_pre/labels")

    model.eval()	# BN and Dropout are different during training and inference
    for i in range(len(video_frames)):      # how many video claps
        frames = video_frames[i]["frames"]
        flow_x = [0] + video_flows[i]["flow_x"]
        flow_y = [0] + video_flows[i]["flow_y"]

        current_block_frame = []
        current_block_flow_x = []
        current_block_flow_y = []

        video_pre_bin = np.zeros(6)

        for i_frame in range(len(frames)):    # how many frames in a video clap
            snippet_pred_bin = np.zeros(6)
            current_block_frame.append(frames[i_frame])
            if i_frame > 0:
                current_block_flow_x.append(flow_x[i_frame])
                current_block_flow_y.append(flow_y[i_frame])

            if i_frame < 14:
                continue

            if i_frame >= 14 and (i_frame + 1) % 5 == 0:    # slide window size: 5, 
                                                            # numbers of input frames into the CNN: 15, 14, 14
                score = getscore(current_block_frame, current_block_flow_x, current_block_flow_y, mean, use_cuda)
      
                current_block_frame = current_block_frame[5:]
                current_block_flow_x = current_block_flow_x[5:]
                current_block_flow_y = current_block_flow_y[5:]
                
                score -= np.max(score)      # deduce the max to avoid overflowing for exp(score)
                p = np.e**score / np.sum(np.e**score)
                snippet_pred_bin[np.argmax(p)] += 1 

                if (i_frame + 1) % (int(block_num)*15) == 0:    # Output a prediction every snippet
                    snippet_pred = np.argmax(snippet_pred_bin)
                    video_pre_bin[snippet_pred] += 1

        video_pre = np.argmax(video_pre_bin)
        label = CATEGORIES.index(video_frames[i]["category"])

        snippet_stat[label] += video_pre_bin

        if video_pre == label:
            correct += 1
            print(video_pre_bin, "%7s" %(np.argmax(video_pre_bin)), "%5s" %(label) )
        else:
            print(video_pre_bin, "%7s" %(np.argmax(video_pre_bin)), "%5s" %(label), "---wrong video predection" )
        
        if i > 0 and i % 20 == 0:
            print("Done %d/%d videos" % (i, len(video_frames)))

print(snippet_stat)
print("%d/%d" % (correct, len(video_frames)), "Accuracy on Videos: %.9f" % (correct / len(video_frames)))

