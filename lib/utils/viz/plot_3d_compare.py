import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import imageio
import textwrap

def plot_3d_motion_overlay(args, figsize=(10, 10), fps=120, radius=4, alt_color=False):
    """
    args = [pred_joints, gt_joints, out_name, title]
    """
    matplotlib.use('Agg')

    pred, gt, out_name, title = args

    pred = pred.copy().reshape(len(pred), -1, 3)
    gt   = gt.copy().reshape(len(gt), -1, 3)

    frame_number = pred.shape[0]
    nb_joints = pred.shape[1]

    # Same chains
    h36m_kinetic_chain = [
        [0, 1, 2, 3, 4],
        [0, 6, 7, 8, 9],
        [0, 11, 12, 13, 14, 15],
        [13, 17, 18, 19],
        [13, 25, 26, 27]
    ]

    # Colors for prediction
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue','darkblue','darkblue','darkblue','darkblue',
              'darkred','darkred','darkred','darkred','darkred']

    colors_alt = ['red','green','black','red','green',
                  'darkgreen','darkgreen','darkgreen','darkgreen','darkgreen',
                  'darkred','darkred','darkred','darkred','darkred']

    # Grey dotted style for GT
    gt_color = (0.5, 0.5, 0.5)
    gt_linestyle = (0, (3, 3))  # dotted line

    # Center + floor align prediction
    MINS = pred.min(axis=0).min(axis=0)
    MAXS = pred.max(axis=0).max(axis=0)
    limits = 1000 if nb_joints == 21 else 2

    height_offset = MINS[1]
    pred[:, :, 1] -= height_offset
    gt[:, :, 1] -= height_offset

    trajec = pred[:, 0, [0, 2]]
    pred[..., 0] -= pred[:, 0:1, 0]
    pred[..., 2] -= pred[:, 0:1, 2]
    gt[..., 0] -= pred[:, 0:1, 0]
    gt[..., 2] -= pred[:, 0:1, 2]

    def update(index):
        fig = plt.figure(figsize=figsize, dpi=300)
        if title is not None:
            wrapped_title = '\n'.join(textwrap.wrap(title, 40))
            fig.suptitle(wrapped_title, fontsize=16)

        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        ax.set_box_aspect([1,1,1])
        ax.grid(False)

        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        # Floor plane
        def plot_xz_plane(minx, maxx, miny, minz, maxz):
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            plane = Poly3DCollection([verts])
            plane.set_facecolor((0.7,0.7,0.7,0.3))
            ax.add_collection3d(plane)

        plot_xz_plane(
            MINS[0] - trajec[index,0],
            MAXS[0] - trajec[index,0],
            0,
            MINS[2] - trajec[index,1],
            MAXS[2] - trajec[index,1]
        )

        # --- PLOT GREY DOTTED GROUND TRUTH UNDERNEATH ---
        for chain in h36m_kinetic_chain:
            ax.plot3D(
                gt[index, chain, 0],
                gt[index, chain, 1],
                gt[index, chain, 2],
                linewidth=3.0,
                color=gt_color,
                linestyle=gt_linestyle,
                alpha=0.8
            )

        # --- PLOT COLORED PREDICTION ON TOP ---
        for i, chain in enumerate(h36m_kinetic_chain):
            ax.plot3D(
                pred[index, chain, 0],
                pred[index, chain, 1],
                pred[index, chain, 2],
                linewidth=4.0,
                color=(colors_alt if alt_color else colors)[i]
            )

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.axis('off')
        plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='raw', dpi=300, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        w, h = fig.get_size_inches() * fig.get_dpi()
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(int(h), int(w), -1)
        buf.close()
        plt.close()
        return arr

    frames = [update(i) for i in range(frame_number)]
    return torch.from_numpy(np.stack(frames, axis=0))

def draw_to_batch_overlay(pred_batch, gt_batch, title_batch=None, outname=None, alt_color=False, figsize=(10,10)):
    out = []
    for i in range(len(pred_batch)):
        frames = plot_3d_motion_overlay(
            [pred_batch[i], gt_batch[i], None, title_batch[i] if title_batch is not None else None],
            alt_color=alt_color,
            figsize=figsize
        )
        out.append(frames)

        if outname is not None:
            imageio.mimsave(outname[i], np.array(frames), fps=20)

    return torch.stack(out, axis=0)
