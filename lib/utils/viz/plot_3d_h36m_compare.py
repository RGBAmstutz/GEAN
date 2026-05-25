import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import imageio
import textwrap

def plot_3d_motion_compare(args, figsize=(10, 10), fps=120, radius=4, alt_color=False):
    matplotlib.use('Agg')
    pred, gt, out_name, title = args
    #pred_data = pred.copy().reshape(len(pred), -1, 3)
    #gt_data = gt.copy().reshape(len(pred), -1, 3)  # Ensure same shape as pred
    #nb_joints = pred_data.shape[1]
    pred_data = np.array(pred).reshape(-1, pred.shape[-1] // 3, 3)
    gt_data = np.array(gt).reshape(-1, gt.shape[-1] // 3, 3)
    nb_joints = pred_data.shape[1]

    h36m_kinetic_chain = [
        [0, 1, 2, 3, 4],  # Hip to Right Foot
        [0, 6, 7, 8, 9],  # Hip to Left Foot
        [0, 11, 12, 13, 14, 15],  # Hip to Head
        [13, 17, 18, 19],  # Thorax to Left Wrist
        [13, 25, 26, 27]  # Thorax to Right Wrist
    ]
    limits = 1000 if nb_joints == 21 else 2
    MINS = pred_data.min(axis=0).min(axis=0)
    MAXS = pred_data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    colors_alt = ['red', 'green', 'black', 'red', 'green',
                  'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen',
                  'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = pred_data.shape[0]

    height_offset = MINS[1]
    pred_data[:, :, 1] -= height_offset
    gt_data[:, :, 1] -= height_offset

    trajec = pred_data[:, 0, [0, 2]]
    pred_data[..., 0] -= pred_data[:, 0:1, 0]
    pred_data[..., 2] -= pred_data[:, 0:1, 2]

    gt_data[..., 0] -= pred_data[:, 0:1, 0]  # Align GT to pred
    gt_data[..., 2] -= pred_data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(False)
            ax.set_box_aspect([1, 1, 1])

        def plot_xz_plane(minx, maxx, miny, minz, maxz):
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.7, 0.7, 0.7, 0.3))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=figsize, dpi=300)
        if title is not None:
            fig.suptitle('\n'.join(textwrap.wrap(title, 40)), fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        init()
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        plot_xz_plane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                      MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        # Plot trajectory
        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue' if not alt_color else 'green')

        # Plot predicted skeleton
        for i, (chain, color) in enumerate(zip(h36m_kinetic_chain, colors if not alt_color else colors_alt)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(pred_data[index, chain, 0], pred_data[index, chain, 1],
                      pred_data[index, chain, 2], linewidth=linewidth, color=color)

        # Plot ground truth skeleton as grey dotted lines
        for chain in h36m_kinetic_chain:
            ax.plot3D(gt_data[index, chain, 0], gt_data[index, chain, 1],
                      gt_data[index, chain, 2], linestyle='dotted', color='grey', linewidth=1.5)

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

        if out_name is not None:
            plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=300, bbox_inches='tight', pad_inches=0)
            io_buf.seek(0)
            width, height = fig.get_size_inches() * fig.get_dpi()
            arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape(int(height), int(width), -1)
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def draw_to_batch_compare(smpl_joints_batch, gt_batch, title_batch=None, outname=None, alt_color=False, figsize=(10,10)):
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        pred_seq = smpl_joints_batch[i]
        gt_seq   = gt_batch[i]
        title    = title_batch[i] if title_batch is not None else None

        out.append(plot_3d_motion_compare([pred_seq, gt_seq, None, title], alt_color=alt_color, figsize=figsize))

        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)

    out = torch.stack(out, axis=0)
    return out