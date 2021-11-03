from anytree import NodeMixin, iterators, RenderTree
import numpy as np
import math
import glob

def Make_Virtual():
    return SwcNode_convert(nid=-1)

class SwcNode_convert(NodeMixin):

    def __init__(self, nid=-1, ntype=0, radius=1, center=[0, 0, 0], parent=None):
        self._id = nid
        self._radius = radius
        self._pos = center
        self._type = ntype
        self.parent = parent

    def is_virtual(self):
        """Returns True iff the node is virtual.
        """
        return self._id < 0

    def is_regular(self):
        """Returns True iff the node is NOT virtual.
        """
        return self._id >= 0

    def get_id(self):
        """Returns the ID of the node.
        """
        return self._id

    def distance(self, tn):
        """ Returns the distance to another node.

        It returns 0 if either of the nodes is not regular.

        Args:
          tn : the target node for distance measurement
        """
        if tn and self.is_regular() and tn.is_regular():
            dx = self._pos[0] - tn._pos[0]
            dy = self._pos[1] - tn._pos[1]
            dz = self._pos[2] - tn._pos[2]
            d2 = dx * dx + dy * dy + dz * dz

            return math.sqrt(d2)

        return 0.0

    def parent_distance(self):
        """ Returns the distance to it parent.
        """
        return self.distance(self.parent)

    def radius(self):
        return self._radius

    def to_swc_str(self):
        return '%d %d %g %g %g %g' % (self._id, self._type, self._pos[0], self._pos[1], self._pos[2], self._radius)

    def get_parent_id(self):
        return -2 if self.is_root else self.parent.get_id()

    def __str__(self):
        return '%d (%d): %s, %g' % (self._id, self._type, str(self._pos), self._radius)


class SwcTree_convert:
    """A class for representing one or more SWC trees.

    For simplicity, we always assume that the root is a virtual node.

    """

    def __init__(self,x_begin=0,y_begin=0,z_begin=0):
        self._root = Make_Virtual()
        self.x_begin = x_begin
        self.y_begin = y_begin
        self.z_begin = z_begin

    def _print(self):
        print(RenderTree(self._root).by_attr("_id"))

    def clear(self):
        self._root = Make_Virtual()

    def is_comment(self, line):
        return line.strip().startswith('#')

    def root(self):
        return self._root

    def regular_root(self):
        return self._root.children

    def node_from_id(self, nid):
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if tn.get_id() == nid:
                return tn
        return None

    def parent_id(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.get_parent_id()

    def parent_node(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.parent

    def child_list(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.children

    def load(self, path):
        self.clear()
        with open(path, 'r') as fp:
            lines = fp.readlines()
            nodeDict = dict()
            for line in lines:
                if not self.is_comment(line):
                    #                     print line
                    data = list(map(float, line.split()))
                    #                     print(data)
                    if len(data) == 7:
                        nid = int(data[0])
                        ntype = int(data[1])
                        pos = data[2:5]
                        pos[0] = (pos[0] + self.x_begin)  # y
                        pos[1] = (pos[1] + self.y_begin)  # x
                        pos[2] = (pos[2] + self.z_begin)  # z
                        radius = data[5]
                        parentId = data[6]
                        tn = SwcNode_convert(nid=nid, ntype=ntype, radius=radius, center=pos)
                        nodeDict[nid] = (tn, parentId)
            fp.close()

        for _, value in nodeDict.items():
            tn = value[0]
            parentId = value[1]
            if parentId == -1:
                tn.parent = self._root
            else:
                parentNode = nodeDict.get(parentId)
                if parentNode:
                    tn.parent = parentNode[0]

    def load_matric(self, path):
        # self.clear()
        with open(path, 'r') as fp:
            lines = fp.readlines()
            # print(len(lines))
            swc_data = np.zeros([len(lines),7])
            k = 0
            nodeDict = dict()
            for line in lines:
                if not self.is_comment(line):
                    data = list(map(float, line.split()))
                    if len(data) == 7:
                        swc_data[k][0] = data[0]
                        swc_data[k][1] = data[1]
                        swc_data[k][2] = data[2]+self.x_begin
                        swc_data[k][3] = data[3]+self.y_begin
                        swc_data[k][4] = data[4]+self.z_begin
                        swc_data[k][5] = data[5]
                        swc_data[k][6] = data[6]
                        k = k+1
            fp.close()
        return swc_data

    def glue_swc(self,path,gap):
        self.clear()
        nodeDict = dict()

        csvx_list_swc = glob.glob(path)
        print('总共发现%s个tif文件' % len(csvx_list_swc))
        num = int(len(csvx_list_swc))

        k = 1
        convert_num = 0
        for path_temp in csvx_list_swc:
            print(path_temp)
            z = int(path_temp[-20:-16])
            x = int(path_temp[-15:-10])
            y = int(path_temp[-9:-4])


            with open(path_temp, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if not self.is_comment(line):
                        data = list(map(float, line.split()))
                        if len(data) == 7:
                            nid = int(k)
                            ntype = int(data[1])
                            pos = data[2:5]
                            pos[0] = (pos[0]-gap + y) * 0.30
                            pos[1] = (pos[1]-gap + x) * 0.30
                            pos[2] = pos[2]-gap + z
                            radius = data[5] * 0.30
                            # pos[0] = pos[0] + y - y_begin
                            # pos[1] = pos[1] + x - x_begin
                            # pos[2] = pos[2] + z - z_begin
                            # radius = data[5]

                            if data[6] == -1:
                                parentId = data[6]
                            else:
                                parentId = data[6] + convert_num
                            tn = SwcNode_convert(nid=nid, ntype=ntype, radius=radius, center=pos)
                            nodeDict[nid] = (tn, parentId)
                            k = k + 1
                            # print(nid,pos,radius,parentId)
                fp.close()
            convert_num = k - 1

        for _, value in nodeDict.items():
            tn = value[0]
            #print(tn)
            parentId = value[1]
            if parentId == -1:
                tn.parent = self._root
            else:
                parentNode = nodeDict.get(parentId)
                if parentNode:
                    tn.parent = parentNode[0]



    def save(self, path):
        with open(path, 'w') as fp:
            niter = iterators.PreOrderIter(self._root)
            for tn in niter:
                if tn.is_regular():
                    fp.write('%s %d\n' % (tn.to_swc_str(), tn.get_parent_id()))
            fp.close()

    def has_regular_node(self):
        return len(self.regular_root()) > 0

    def node_count(self, regular=True):
        count = 0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    count += 1
            else:
                count += 1

        return count

    def parent_distance(self, nid):
        d = 0
        tn = self.node(nid)
        if tn:
            parent_tn = tn.parent
            if parent_tn:
                d = tn.distance(parent_tn)

        return d

    def scale(self, sx, sy, sz, adjusting_radius=True):
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            tn.scale(sx, sy, sz, adjusting_radius)

    def length(self):
        niter = iterators.PreOrderIter(self._root)
        result = 0
        for tn in niter:
            result += tn.parent_distance()

        return result

    def radius(self, nid):
        return self.node(nid).radius()


if __name__ == '__main__':

    tree_dir = 'C:/Users/78286/Desktop/18253/18253.swc'
    tree_Brain.load(tree_dir)

    Brain_dir = 'C:/Users/78286/Desktop/18253/18253_convert.swc'
    tree_Brain.save(Brain_dir)
