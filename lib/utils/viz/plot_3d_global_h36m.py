import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import imageio
import textwrap

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4, alt_color=False):
    matplotlib.use('Agg')
    joints, out_name, title = args
    data = joints.copy().reshape(len(joints), -1, 3)
    nb_joints = data.shape[1]

    h36m_kinetic_chain = [
        [0, 1, 2, 3, 4],  # Hip to Right Foot
        [0, 6, 7, 8, 9],  # Hip to Left Foot
        [0, 11, 12, 13, 14, 15],  # Hip to Head
        [13, 17, 18, 19],  # Thorax to Left Wrist
        [13, 25, 26, 27]  # Thorax to Right Wrist
    ]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    colors_alt = ['red', 'green', 'black', 'red', 'green',
                  'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen',
                  'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(False)
            ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio

        def plot_xz_plane(minx, maxx, miny, minz, maxz):
            # Plot the grey plane under the model's feet
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.7, 0.7, 0.7, 0.3))  # Grey with some transparency
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=300)
        if title is not None:
            wrapped_title = '\n'.join(textwrap.wrap(title, 40))
            fig.suptitle(wrapped_title, fontsize=16)
        ax = fig.add_subplot(111, projection='3d')

        init()

        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        # Plot the grey plane under the feet
        plot_xz_plane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                      MAXS[2] - trajec[index, 1])

        # Plot the trajectory line (if applicable)
        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue' if not alt_color else 'green')

        # Plot the 3D skeleton
        for i, (chain, color) in enumerate(zip(h36m_kinetic_chain, colors if not alt_color else colors_alt)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.axis('off')

        # Control layout and whitespace tightly
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

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, alt_color=False, figsize=(10,10)):
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None], alt_color=alt_color, figsize=figsize))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out
