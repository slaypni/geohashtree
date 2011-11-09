#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Purpose:     Class library for storing and operating geohash-indexed data.
# Copyright:   (c) Kazuaki Tanida 2011
# Licence:     MIT License
#-------------------------------------------------------------------------------


import math
import bisect


class Node(object):
    __slots__ = ['parent', 'children', 'value', 'depth'] # self.children = [left, right]

    def __init__(self, parent=None, children=None, depth=None, value=None):
        object.__init__(self)
        self.parent= parent
        self.children= children
        self.value= value
        self.depth= depth


    def getPath(self):
        r = []
        node = self
        while node.parent != None:
            i = 0 if node.parent.children[0] == node else 1
            r.append(i)
            node = node.parent
        return r[::-1]


    def getPathStr(self):
        return ''.join(str(i) for i in self.getPath())


    def encodeSuccinct(self, doValues=False):
        def _encodeSuccinct(node, bits, data=None):
            if node.children == None:
                bits.append(0)
                if data != None:
                    data.append(node.value)
            else:
                bits.append(1)
                _encodeSuccinct(node.children[0], bits, data)
                _encodeSuccinct(node.children[1], bits, data)
        bits = []
        data = [] if doValues else None
        _encodeSuccinct(self, bits, data)
        if doValues:
            return _bit2str(bits), data
        else:
            return _bit2str(bits)


    @classmethod
    def decodeSuccinct(cls, bitstr, data=None):
        def _decodeSuccinct(bits, data=None, _depth=0, _parent=None):
            n = cls(depth=_depth, parent=_parent)
            b = bits.pop(0)
            if b == 0:
                if data:
                    n.value = data.pop(0)
            else:
                left = _decodeSuccinct(bits, data, _depth+1, n)
                right = _decodeSuccinct(bits, data, _depth+1, n)
                n.children = [left, right]
            return n
        return _decodeSuccinct([b for b in _str2bit(bitstr)], data)


    def printS(self):
        r = ''
        node = self
        s = []
        while node:
            r += '('
            if node.value != None:
                r += str(len(node.value))
            if node.children != None:
                lc = node.children[0]
                rc = node.children[1]
                s.append(rc)
                s.append(lc)
            if s:
                next = s.pop()
                c = node.depth - next.depth + 1
            else:
                next = None
                c = node.depth + 1
            r += ')' * c
            node = next
        print r


    def getSibling(self):
        if self.parent == None:
            return None
        lc = self.parent.children[0]
        rc = self.parent.children[1]
        return lc if self is not lc else rc


    def getRoot(self):
        node = self
        while node.parent != None:
            node = node.parent
        return node


    def setChildren(self, left, right):
        self.children = [left, right]


    def removeChildren(self):
        self.children = None


    def isLeaf(self):
        return True if self.children == None else False


    def __iter__(self):
        node = self
        s = []
        while node:
            yield node
            if node.isLeaf() == False:
                s.extend(node.children[::-1])
            node = s.pop() if s else None


class BinaryTree(object):
    def _getDefaultValue(self):
        return self.defaultValueFactory() if self.defaultValueFactory else None
    defaultValue = property(_getDefaultValue)


    def __init__(self, NodeCls=Node, defaultValueFactory=None, root=None):
        self.NodeCls = NodeCls
        self.defaultValueFactory = defaultValueFactory
        self.root = self.NodeCls(depth=0, value=self.defaultValue) if root == None else root


    def _addChildren(self, parent):
        if parent.isLeaf() == False:
            return
        lc = self.NodeCls(parent=parent, depth=parent.depth+1, value=self.defaultValue)
        rc = self.NodeCls(parent=parent, depth=parent.depth+1, value=self.defaultValue)
        parent.setChildren(lc, rc) # parent.children = [lc, rc]


    def printS(self):
        self.root.printS()


    def __iter__(self, node=None):
        node = self.root if node == None else node
        s = []
        while node:
            if node.isLeaf() == True: #if node.value != None: # the later discription causes trouble when using ght4gae, since a parent's value include the children's one in that case
                yield node.value
            else:
                s.extend(node.children[::-1])
            node = s.pop() if s else None



class GeohashTree(BinaryTree):
    def __init__(self, max_val_len=1, max_val_len_depth_coef=0.5, max_depth=60, NodeCls=Node, root=None):
        BinaryTree.__init__(self, defaultValueFactory=dict, NodeCls=NodeCls, root=root)
        if root == None:
            self.root.value = self.defaultValueFactory()
        self.max_val_len = max_val_len
        self.max_val_len_depth_coef = max_val_len_depth_coef
        self.max_depth= max_depth


    def _getMaxValueLength(self, node): # bucket size
        return self.max_val_len + math.ceil(node.depth * self.max_val_len_depth_coef)


    def _addValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        node.value[hashcode] = val
        if len(node.value) > self._getMaxValueLength(node) and node.depth < self.max_depth:
            self._addChildren(parent=node)


    #@override
    def _addChildren(self, parent):
        BinaryTree._addChildren(self, parent)
        for (hashcode, val) in parent.value.items():
            self._addValue(hashcode, val, node=parent)
        parent.value = None


    def getValue(self, hashcode):
        node = self.getLeaf(hashcode)
        return node.value.get(hashcode)


    def getLeaf(self, hashcode): # get the bucket which a given hashcode belongs to
        return self._getLeafFromNode(hashcode, self.root)


    def _getLeafFromNode(self, hashcode, node, geobits=None):   # given geobits, hashcode is not necessary
        if node.children == None:
            return node
        else:
            geobits = [c for c in yieldGeobit(hashcode)] if geobits == None else geobits
            if len(geobits) <= node.depth:
                return node
            i = geobits[node.depth]
            return self._getLeafFromNode(hashcode, node.children[i], geobits)


    def _removeChildren(self, parent):
        parent.removeChildren()



    def removeMatchedValue(self, hashcode, val):
        node = self.getLeaf(hashcode)
        if node.value.get(hashcode) == val:
            del node.value[hashcode]
            self._removeNode(node)


    def _removeNode(self, node): # remove arg node if the parent can merge it
        while True:
            if node == None:
                break
            if node.parent == None: # root node
                break
            sib = node.getSibling()
            if sib.isLeaf() == False: # without this "if" statement, fail case: (((2)(2))(1)), safe case: ((2)((2)(1)))
                break
            if len(node.value) + len(node.getSibling().value) > self._getMaxValueLength(node.parent):
                break
            node.parent.value = self.defaultValueFactory() # if node.parent.value == None else node.parent.value    # the later if statement may cause problem when using ght4gae
            node.parent.value.update(node.value)
            node.parent.value.update(node.getSibling().value)
            self._removeChildren(node.parent)
            node = node.parent


    def removeValue(self, hashcode, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        del node.value[hashcode]
        self._removeNode(node)


    # replace corresponding data, if given hashcode already existed. else, just add given data.
    def setValue(self, hashcode, val):
        self._addValue(hashcode, val)


    # nrange indicate the range of neighbor. ex. [0,1,2,3] includes nodes inside 3 times distance (1,2,3) and itself (0).
    def getNeighborLeaves(self, hashcode, nrange=[0, 1], depth=-1):
        geobits = [c for c in yieldGeobit(hashcode)]
        node = self._getLeafFromNode(hashcode, self.root, geobits=geobits)
        nodebits = geobits[:node.depth if depth == -1 else depth]
        lat, lon, lat_len, lon_len = _decode2int(nodebits)
        depth_mask = 0
        for i in range(node.depth):
            depth_mask <<= 1
            depth_mask += 1
        nr = [-1 * d for d in range(max(nrange)+1)[::-1] if d != 0] + range(max(nrange)+1)
        pos = set(((lat + y) & depth_mask, (lon + x) & depth_mask) for x in nr for y in nr if (abs(x) in nrange) or (abs(y) in nrange))
        nnodes = set()
        for (y, x) in pos:
            geobits=[b for b in _int2bit(_getMortonNumber(y, x, lat_len, lon_len), lat_len+lon_len)]
            nnodes.update(node for node in self._getLeafFromNode(hashcode, self.root, geobits=geobits) if node.children == None)
        return nnodes


    def _getNeighborPaths(self, hashcode, scale=1, depth=-1, include_hash_path=False):
        geobits = [c for c in yieldGeobit(hashcode)]
        node = self._getLeafFromNode(hashcode, self.root, geobits=geobits)
        nodebits = geobits[:node.depth if depth == -1 else depth]
        lat, lon, lat_len, lon_len = _decode2int(nodebits)
        start_morton_bits_list = []
        if scale > 0:
            depth_mask = 0
            for i in range(node.depth):
                depth_mask <<= 1
                depth_mask += 1
            lats = set((lat + n) & depth_mask for n in [-1, 0, 1])
            lons = set((lon + n) & depth_mask for n in [-1, 0, 1])
            pos = [(y, x) for x in lons for y in lats]
            pos.remove((lat, lon))
            for (y, x) in pos:
                start_morton_bits_list.append( [b for b in _int2bit(_getMortonNumber(y, x, lat_len, lon_len), lat_len+lon_len)][:lat_len+lon_len-scale+1] )
        if include_hash_path == True:
            start_morton_bits_list.append( [b for b in _int2bit(_getMortonNumber(lat, lon, lat_len, lon_len), lat_len+lon_len)][:lat_len+lon_len] )
        return start_morton_bits_list


    # find number of (scale)'s nodes at each eight directions (if node's area are smaller than a given scale)
    def getSurroundingLeaves(self, hashcode, scale=1):
        node = self.getLeaf(hashcode)
        targen_morton_bits = node.getPath() #[b for b in _int2bit(_getMortonNumber(lat, lon, lat_len, lon_len), lat_len+lon_len)]
        surroundings = []
        for start_morton_bits in self._getNeighborPaths(hashcode, scale=scale):
            surroundings.extend( self._getNearestNeighborNodes(start_morton_bits, targen_morton_bits, target_node=node, k=scale) )
        return list(set(surroundings))


    # given a start node, find the leaf nearest to the target (or k-NN leaves)
    def _getNearestNeighborNodes(self, start_morton_bits, targen_morton_bits, start_node=None, target_node=None, k=1, r=None, q=None):
        r = [] if r == None else r
        if q == None:
            node = self._getLeafFromNode(None, self.root, start_morton_bits) if start_node == None else start_node
            q = []
        else:
            if len(q) == 0:
                return r
            (distance, start_morton_bits, node) = q.pop(0)
        if node.children == None:
            r.append(node)
            return r
        target_node = self._getLeafFromNode(None, self.root, targen_morton_bits) if target_node == None else target_node
        d = node.depth - target_node.depth
        _targen_morton_bits, _start_morton_bits = targen_morton_bits, start_morton_bits
        if d >= 0:
            _targen_morton_bits = targen_morton_bits[:target_node.depth] + [0] * (d + 1)
        elif d < 0:
            _start_morton_bits = start_morton_bits[:node.depth] + [0] * (-1 * d - 1)
        ty, tx, ty_len, tx_len = _decode2int(_targen_morton_bits)
        for b in [0, 1]:
            ny, nx, ny_len, nx_len = _decode2int(_start_morton_bits+[b])
            n = (math.hypot(ty-ny, tx-nx), _start_morton_bits+[b], node.children[b])
            q.insert(bisect.bisect(q, n), n)
        while len(r) < k and len(q) > 0:
            self._getNearestNeighborNodes(None, targen_morton_bits, target_node=target_node, k=k, r=r, q=q)
        return r


    def getNearestNeighborItems(self, hashcode, k=1):
        ty, tx, ty_len, tx_len = _decode2int(yieldGeobit(hashcode))
        tn = self.getLeaf(hashcode)
        tm = tn.getPath()
        tny, tnx, tny_len, tnx_len = _decode2int(tm)
        nodes =set()
        nodes.add(tn)
        cands = []
        visited = set()

        while nodes:
            _nodes = nodes
            nodes =set()
            for n in _nodes:
                for hc in n.value:
                    y, x, y_len, x_len = _decode2int(yieldGeobit(hc))
                    y = (y << (ty_len - y_len)) if ty_len > y_len else (y >> (y_len - ty_len))
                    x = (x << (tx_len - x_len)) if tx_len > x_len else (x >> (x_len - tx_len))
                    d = (math.hypot(ty-y, tx-x), n, (hc, n.value[hc]))
                    cands.insert(bisect.bisect(cands, d), d)
                cands = cands[:k]

                visited.add(n)
                thr = cands[-1][0] if len(cands) == k else 1e308
                _y, _x, y_len, x_len = _decode2int(n.getPath())
                for (a, b) in [(a, b) for a in [-1, 0, 1] for b in [-1, 0, 1]]: # if any vertex of neighbor regions closer than the most far candidate point, add the region's node to the candidate node's list (nodes)
                    y += _y + a
                    x += _x + b
                    ly = (y << (ty_len - y_len)) if ty_len > y_len else (y >> (y_len - ty_len))
                    lx = (x << (tx_len - x_len)) if tx_len > x_len else (x >> (x_len - tx_len))
                    ry = _bit2int([b for b in _int2bit(y, y_len)] + [1] * (ty_len - y_len)) if ty_len > y_len else ly
                    rx = _bit2int([b for b in _int2bit(x, x_len)] + [1] * (tx_len - x_len)) if tx_len > x_len else lx
                    rect = [(ly, lx), (ry, lx), (ly, rx), (ry, rx)]
                    if any( (math.hypot(r[0]-ty, r[1]-tx) < thr) for r in rect ):
                        start_morton_bits = [b for b in _int2bit(_getMortonNumber(y, x, y_len, x_len), y_len+x_len)]
                        neighbor = self._getNearestNeighborNodes(start_morton_bits, tm, k=1)[0]
                        if neighbor not in visited:
                            nodes.add(neighbor)
                            visited.add(neighbor)

        #return cands
        return [(hc, val) for (dist, node, (hc, val)) in cands]



    def __getitem__(self, key):
        r = self.getValue(key)
        if r != None:
            return r
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.setValue(key, value)

    def __delitem__(self, key):
        self.removeValue(key)

    def __iter__(self, node=None):
        for value in BinaryTree.__iter__(self):
            for hashcode in value:
                yield hashcode

    def __len__(self):
        i = 0
        for a in self.__iter__():
            i += 1
        return i

    def keys(self):
        return [k for k in self]

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def __contains__(self, key):
        n = self.getLeaf(key)
        return (key in n.value)


# GeohashTree cannot correctly manage data which has the existing geohash. for this case, GeoPileTree can be used.
# GeoPileTree stores a number of data which have the same geohash togather. ex. self.value = {geo1: [val1, val2], geo3: [val3], ...}
# however, GeoPileTree does not allow setting data using "=".

class GeoPileTree(GeohashTree):

    def addValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        v = node.value.get(hashcode)
        if v == None:
            node.value[hashcode] = [val]
            if len(node.value) > self._getMaxValueLength(node) and node.depth < self.max_depth:
                self._addChildren(parent=node)
        else:
            v.append(val)

    #_addValue = GeohashTree._addValue

    #@override
    def setValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        v = node.value.get(hashcode)
        #print val.text
        if v != None:
            node.value[hashcode] = [val]
        else:
            self.addValue(hashcode, val, node=node)

    #@override
    def removeMatchedValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        v = node.value.get(hashcode)
        if v != None:
            try:
                v.remove(val)
                if v == []:
                    self.removeValue(hashcode, node=node)
            except ValueError:
                pass

    __setitem__ = None


# -- start: geohash operations -- #


_base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_base32_map = dict((c, i) for (i, c) in enumerate(_base32))


def yieldGeobit(hashcode):
    for i in hashcode:
        t = _base32_map[i]
        for j in range(5):
            r = (t & 0x10) >> 4 # r = (t & 0b10000) >> 4
            yield r
            t <<= 1


def _decode2int(giobits):
    lon = lat = 0
    lat_len = lon_len = 0
    for i, b in enumerate(giobits):
        if i %2 == 0:
            lon <<= 1
            lon += b
            lon_len += 1
        else:
            lat <<= 1
            lat += b
            lat_len += 1
    return lat, lon, lat_len, lon_len


def bit2geohash(bits):
    bits = [b for b in bits]
    s = ''
    while bits != []:
        n = 0
        for i in range(5)[::-1]:
            if len(bits) == 0:
                break
            b = bits.pop(0)
            n += b << i
        s += _base32[n]
    return s


def _getMortonNumber(y, x, y_len, x_len):
    r = 0
    for lon, lat in zip(_int2bit(x, x_len),_int2bit(y, y_len)):
        r <<= 2
        r += (lon << 1) + lat
    if (x_len - y_len) == 1:
        r <<= 1
        r += x & 1
    return r


def _int2bit(num, length):
    for b in range(length-1, -1, -1):
        yield (num >> b) & 1


def _bit2int(bits):
    r = 0
    for b in bits:
        r <<= 1
        r += b
    return r


def _bit2str(bits):
    bs = ''
    while bits:
        n = sum([((bits[i] if len(bits) > i else 0) << 7) >> i for i in range(8)])
        bs = bs + chr(n)
        bits = bits[8:]
    return bs


def _str2bit(bitstr):
    for c in bitstr:
        n = ord(c)
        for b in [(n >> i) & 1 for i in range(8)[::-1]]:
            yield b


# -- end: geohash operations -- #


import random
try:
    import geohash  # http://code.google.com/p/python-geohash/
except:
    pass

def test(n=10, scale=1, k=5, GeohashTreeCls=GeohashTree):
    locations = [(random.uniform(-90.0, 90.0), random.uniform(-180.0, 180.0)) for i in range(n)]
    g = GeohashTreeCls() # GeoPileTree()
    for p in locations:
        h = geohash.encode(*p)
        #print h
        #g[h] = random.randint(0, 1)
        g.setValue(h,random.randint(0, 9)) # "=" doesn't be used when g = GeoPileTree()
    g.printS()
    print (g.keys()[n/2], g[g.keys()[n/2]])
    _test_surroundings(g.keys()[n/2], g, scale)
    _test_nn_value(g.keys()[n/2], g, k)
    for p in locations:
        h = geohash.encode(*p)
        #g.removeMatchedValue(h, random.randint(0, 0))
        #del g[h]
    g.printS()
    return g


def _test_surroundings(hashcode, g, k=1):
    #s = g.getSurroundingLeaves(hashcode, scale=k)
    s = g.getNeighborLeaves(hashcode, nrange=range(k+1))
    _xy = _decode2int(g.getLeaf(hashcode).getPath())
    print g.getLeaf(hashcode).getPath(), (_xy[0], _xy[1])
    for n in s:
        xy = _decode2int(n.getPath())
        a = xy[0] << (_xy[2]-xy[2]) if _xy[2]>xy[2] else xy[0] >> (xy[2]-_xy[2])
        b = xy[1] << (_xy[3]-xy[3]) if _xy[3]>xy[3] else xy[1] >> (xy[3]-_xy[3])
        print (n.getPath(), (a, b), n.value)
    return s


def _test_nn_value(hashcode, g, k=5):
    vs = g.getNearestNeighborItems(hashcode, k=k)
    print vs
    return vs


