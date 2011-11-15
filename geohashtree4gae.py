# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Purpose:     geohashtree for storing with google app engine.
# Copyright:   (c) Kazuaki Tanida 2011
# Licence:     MIT License
#-------------------------------------------------------------------------------
# - How to use -
# 1. load.     ex. geotree = StoredGeoPileTree.load()
# 2. operate.  ex. geotree.addValue(hashcode, message)
# 3. commit.   ex. geotree.commit()
#-------------------------------------------------------------------------------

try:
    from google.appengine.ext import db
    from google.appengine.api import memcache
except:
    pass

import datetime
import logging

import geohashtree as ght

"""
class SimpleNode(object):   # used for memcache and db
    __slots__ = ['children']

    @classmethod
    def convertBunch(bunch_root):
        simple_bunch_root

        return simple_bunch_root
"""


class GeoMessage(object):
    __slots__ = ['_hashcode', '_text', '_name', '_mail', '_date', 'key', 'modified']

    def __init__(self, hashcode=None, text=None, name=None, mail=None, date=None):
        self._hashcode = hashcode
        self._text = text
        self._name = name
        self._mail = mail
        self._date = date
        self.key = None
        self.modified = True

    @classmethod
    def load(cls, value_db):
        m = cls()
        m._hashcode = value_db.hashcode
        m._text = value_db.text
        m._name = value_db.name
        m._mail = value_db.mail
        m._date = value_db.date
        m.key = value_db.key()
        m.modified = False
        return m


    def _getGeohash(self):
        return self._hashcode

    def _setGeohash(self, data):
        self._hashcode = data
        self.modified = True

    hashcode = property(_getGeohash, _setGeohash)

    def _getText(self):
        return self._text

    def _setText(self, data):
        self._text = data
        self.modified = True

    text = property(_getText, _setText)

    def _getName(self):
        return self._name

    def _setName(self, data):
        self._name = data
        self.modified = True

    name = property(_getName, _setName)

    def _getMail(self):
        return self._mail

    def _setMail(self, data):
        self._mail = data
        self.modified = True

    mail = property(_getMail, _setMail)

    def _getDate(self):
        return self._date

    def _setDate(self, data):
        self._date = data
        self.modified = True

    date = property(_getDate, _setDate)


    # store a message into db
    def commit(self):
        if self.modified == False:
            return
        if self.key != None:
            value_db = ValueDB.get(self.key)
            value_db.hashcode = self._hashcode,
            value_db.text = self._text,
            value_db.name = self._name,
            value_db.mail = self._mail,
            value_db.date = self._date
            value_db.put()
        else:
            value_db = ValueDB(
                hashcode = self._hashcode,
                text = self._text,
                name = self._name,
                mail = self._mail,
                date = self._date
            )
            value_db.put()
            self.key = value_db.key()
        self.modified = False


# the ValueDB entity represent the val of the hashcode. therefore, one bucket might have multiple entity.
# futhermore, each hashcode might have multiple val. therefore, one hashcode might have multiple entity.
# namely, a ValueDB entity represents one message.
class ValueDB(db.Model):
    hashcode = db.StringProperty(required=True)
    text = db.TextProperty(required=True)
    name = db.StringProperty()
    date = db.DateTimeProperty(required=True)
    mail = db.StringProperty()

    @classmethod
    def new(cls, hashcode, text, name=None, mail=None, date=None):
        value_db = ValueDB(
            hashcode = hashcode,
            text = text,
            date = datetime.datetime.now() if date == None else date
        )
        value_db.name = name
        value_db.mail = mail
        return value_db

    # get values below the given path
    @classmethod
    def getValues(cls, path_bits, max_depth=60):
        value = {}
        bits = [b for b in path_bits]
        if len(bits) > 0:
            min_hash = ght.bit2geohash(bits)
            #max_hash = ght.bit2geohash(ght._int2bit(ght._bit2int(bits) + 1, len(bits) + (1 if all(bits) else 0)))
            #max_hash = min_hash[:-1] + chr(ord(min_hash[-1])+1)
            max_hash = ght.bit2geohash(bits + [1]*(max_depth-len(bits)))
            q = ValueDB.gql('WHERE hashcode >= :min_hash AND hashcode <= :max_hash', min_hash=min_hash, max_hash=max_hash)
        else:
            q = ValueDB.all()
        for value_db in q:
            value.setdefault(value_db.hashcode, []).append(GeoMessage.load(value_db))
        return value


class StoredNode(ght.Node):
    __slots__ = ['_value', '_is_value_loaded']

    def __init__(self, parent=None, children=None, depth=None, value=None):
        self.parent = parent
        self.children = children
        self.depth = depth
        self._value = value
        self._is_value_loaded = False if value == None else True


    def _getValues(self, see_memcache=True):
        if self._is_value_loaded == True:
            return self._value
        else:
            self._is_value_loaded = True
            self._value = memcache.get('_' + self.getPathStr(), namespace=self.getRoot().value_namespace) if see_memcache else None
            if self._value != None:
                return self._value
            self._value = (self.getRoot().value_db_cls).getValues(self.getPath())
            memcache.set('_' + self.getPathStr(), self._value, namespace=self.getRoot().value_namespace)
            return self._value

    def _setValues(self, value):
        self._is_value_loaded = True
        self._value = value

    value = property(_getValues, _setValues)

    """
    def addValue(self, hashcode, message):
        pass
    """

    #@override
    def removeChildren(self):
        for i in ('0', '1'):
            memcache.delete('_' + self.getPathStr() + i, namespace=self.getRoot().value_namespace)
        self.children = None


    # store self.value into memcache. store modified messages into db.
    def commit(self):
        if self._value != None:
            for messages in self._value.values():
                for message in messages:
                    message.commit()
            memcache.set('_' + self.getPathStr(), self._value, namespace=self.getRoot().value_namespace)
        else:
            memcache.delete('_' + self.getPathStr(), namespace=self.getRoot().value_namespace)
        self._is_value_loaded = True




# root has extra member variables such as memcache's namespace. however, StoredNode has __slots__.
# StoredRootNode is used as the root of StoredNode's tree.
class StoredRootNode(StoredNode):

    @classmethod
    def loadRoot(cls, root_node_db_key_name='StoredNode_root', root_node_key='root', node_namespace='StoredNode', value_namespace='StoredValue', value_db_cls=ValueDB, see_memcache=True):
        bitstr = memcache.get(root_node_key, namespace=node_namespace) if see_memcache else None
        if bitstr != None:
            _root = StoredNode.decodeSuccinct(bitstr)
            bunch_db = None
        else:
            bunch_db = BunchDB.get_or_insert(root_node_db_key_name, bitstr=chr(0))
            memcache.set(root_node_key, bunch_db.bitstr, namespace=node_namespace)
            _root = bunch_db.getBunch()
        root = cls(parent=_root.parent, children=_root.children, depth=_root.depth, value=None)
        if root.isLeaf() == False:
            for c in root.children:
                c.parent = root
        root.root_node_db_key_name = root_node_db_key_name
        root.root_node_key = root_node_key
        root.node_namespace = node_namespace
        root.value_namespace = value_namespace
        root.value_db_cls = value_db_cls
        root.bunch_db = bunch_db
        return root


    # store the structure of the bunch (value is not included)
    def commitBunch(self):
        if self.bunch_db == None:
            self.bunch_db = BunchDB.get_or_insert(self.root_node_db_key_name, bitstr=chr(0))
        self.bunch_db.putBunch(self)
        memcache.set(self.root_node_key, self.bunch_db.bitstr, namespace=self.node_namespace)



class BunchDB(db.Model):
    bitstr = db.BlobProperty(required=True)

    def putBunch(self, root):
        self.bitstr = root.encodeSuccinct()
        self.put()

    def getBunch(self):
        return StoredNode.decodeSuccinct(self.bitstr)


class StoredGeoPileTree(ght.GeoPileTree):

    def __init__(self, max_val_len=1, max_val_len_depth_coef=0.5, max_depth=60, NodeCls=StoredNode, root=None):
        ght.GeoPileTree.__init__(self, max_val_len, max_val_len_depth_coef, max_depth, NodeCls, root)
        self.committing_nodes = set() # list of nodes should be commited
        self.is_tree_committing = False # whether tree should be commited (whether the structure is changed)

    @classmethod
    def load(cls, tree_db_key_name='SGPT_Primary', memcache_key='Primary', tree_namespace='SGPT', value_namespace='StoredValue', see_memcache=True):
        root = StoredRootNode.loadRoot(
            root_node_db_key_name = tree_db_key_name,
            root_node_key = memcache_key,
            node_namespace = tree_namespace,
            value_namespace = value_namespace,
            see_memcache = see_memcache
        )
        return StoredGeoPileTree(NodeCls=StoredNode, root=root)

    #@override
    def _addValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        self.committing_nodes.add(node)
        ght.GeoPileTree._addValue(self, hashcode, val, node=node)

    #@override
    def _addChildren(self, parent):
        ght.GeoPileTree._addChildren(self, parent)
        self.committing_nodes.add(parent)
        for c in parent.children:
            self.committing_nodes.add(c)
        self.is_tree_committing = True

    #@override
    def _removeChildren(self, parent):
        for c in parent.children:
            self.committing_nodes.discard(c)
        ght.GeoPileTree._removeChildren(self, parent)
        self.is_tree_committing = True


    #@override
    def addValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        self.committing_nodes.add(node)
        ght.GeoPileTree.addValue(self, hashcode, val, node)


    #@override
    def setValue(self, hashcode, val, node=None):
        for e in ValueDB.gql('WHERE hashcode = :hashcode', hashcode=hashcode):
            e.delete()
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        self.committing_nodes.add(node)
        ght.GeoPileTree.setValue(self, hashcode, val, node)


    #@override
    def removeMatchedValue(self, hashcode, val, node=None):
        node = self.getLeaf(hashcode) if node == None else self._getLeafFromNode(hashcode, node)
        self.committing_nodes.add(node)
        ght.GeoPileTree.removeMatchedValue(self, hashcode, val, node)


    # store all changes
    def commit(self):
        if self.is_tree_committing == True:
            self.root.commitBunch()
            self.is_tree_committing = False
        for node in self.committing_nodes:
            node.commit()
        self.committing_nodes = set()




"""
class _StoredNode(object):
    __slots__ = ['_node', '_is_children_loaded', '_is_value_loaded, _namespace']

    modified = []

    stored_bunch_interval = 5

    def __init__(self, node=None, parent=None, children=None, depth=None, value=None, namespace='StoredNode'):
        self._is_children_loaded = self._is_value_loaded = False
        self._namespace = namespace
        if node:
            self._node = node
        else:
            self._node = ght.Node(parent=parent, children=children, depth=depth, value=value)
        if self._node.value != None:
            self._is_value_loaded = True


    def _getParent(self):
        pass

    parent = property(_getParent, _setParent)

    def _getChildren(self):
        if self._is_children_loaded == True:
            return self._node.children
        else:
            self._loadChildren()


    children = property(_getChildren, _setChildren)

    def _loadChildren(self):
        if self.is_children_loaded == True:
            return self.children
        for i in [0, 1]:
            bunch_root = memcache.get(self.getPathStr() + str(i), namespace=self._namespace)
            q = [(self, i, bunch_root)]
            while q:    # recompose Node_Bunch -> StoredNode_Bunch
                p, j, n = q.pop()
                sn = StoredNode(node=n, namespace=self._namespace)
                sn._is_children_loaded = True if (sn._node.depth % self.stored_bunch_interval != 0) else False
                if n.children != None:
                    q.extend([(sn, k, c) for k, c in enumerate(n.children)])
                p._node.children[i] = n
                n._node.parent = p
        return self.children


    def getPathStr(self):
        return self._node.getPathStr()


    def getPath(self):
        return self._node.getPath()
"""



