from oil.utils.utils import export
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

@export
class Animation(object):
    def __init__(self, qt,body=None):
        """ [qt (T,n,d)"""
        self.qt = qt
        T,n,d = qt.shape
        assert d in (2,3), "too many dimensions for animation"
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1],projection='3d') if d==3 else self.fig.add_axes([0, 0, 1, 1])
        xyzmin = qt.min(dim=0)[0].min(dim=0)[0]
        xyzmax = qt.max(dim=0)[0].max(dim=0)[0]
        delta = xyzmax-xyzmin
        lower = xyzmin-.1*delta; upper = xyzmax+.1*delta
        self.ax.set_xlim((lower[0],upper[0]))
        self.ax.set_ylim((lower[1],upper[1]))
        if d==3: self.ax.set_zlim((lower[2],upper[2]))
        if d!=3: self.ax.set_aspect("equal")
        empty = d*[[]]
        self.objects = {
            'pts':sum([self.ax.plot(*empty, "o", ms=6) for i in range(n)], []),
            'traj_lines':sum([self.ax.plot(*empty, "-") for _ in range(n)], []),
        }
        
    def init(self):
        empty = 2*[[]]
        for obj in self.objects.values():
            for elem in obj:
                elem.set_data(*empty)
                if self.qt.shape[-1]==3: elem.set_3d_properties([])
        return sum(self.objects.values(),[])

    def update(self, i=0):
        T,n,d = self.qt.shape
        for j in range(n):
            xyz = self.qt[i - 50 if i > 50 else 0 : i + 1,j,:]
            self.objects['traj_lines'][j].set_data(*xyz[...,:2].T)
            if d==3: self.objects['traj_lines'][j].set_3d_properties(xyz[...,2].T.data.numpy())
            self.objects['pts'][j].set_data(*xyz[-1:,...,:2].T)
            if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T.data.numpy())
        #self.fig.canvas.draw()
        return sum(self.objects.values(),[])

    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[0],
                    interval=33,init_func=self.init,blit=True,).to_html5_video()