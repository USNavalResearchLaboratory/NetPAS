# heap with update that subclasses a dictionary

# Python dictionary references
# v2: https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
# v3: https://docs.python.org/3/library/stdtypes.html#mapping-types-dict

# dictionary subclassing pointers
# https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
# https://stackoverflow.com/questions/21309374/how-to-make-a-sorted-dictionary-class

from itertools import chain

import sys

# Python version variables
# https://github.com/benjaminp/six/blob/master/six.py
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

_RaiseKeyError = object()

if PY2:
    items = 'iteritems'
if PY3:
    items = 'items'

class heapqup(dict):
    __slots__ = ('_gt', '_heapq', '_pos', '_inv', 'tiebreak')
    # instance variables:
    #   (inherited map): map from keys to their priorities
    #   _gt: greater-than comparison function between two keys used to order heap
    #   _heapq: a list used to store priorities in heap order
    #   _pos: a map from priorities to heap positions (indices in _heapq)
    #   _inv: a map from priorities to a list or set of keys with that priority
    #   tiebreak: a function to select a key from a set of keys with equal priority, or one of the objects FIFO or LIFO
    #       (by default, None, meaning that selection is left to Python ordering)

    (FIFO, LIFO) = (object(), object())

    def _isordered(self):
        return self.tiebreak in (self.FIFO, self.LIFO)

    @staticmethod
    def _parent(idx):
        return (idx+1)//2 - 1

    @staticmethod
    def _leftchild(idx):
        return (idx+1)*2 - 1

    @staticmethod
    def _rightchild(idx):
        return (idx+1)*2

    def _swap(self, x, y, remap=True):
        vx = self._heapq[x]
        vy = self._heapq[y]
        self._heapq[x] = vy
        self._heapq[y] = vx
        if remap:
            self._pos[vx] = y
            self._pos[vy] = x

    def _topdown(self, idx, remap=True):
        left = self._leftchild(idx)
        right = left+1
        left_bad = left < len(self._heapq) and self._gt(self._heapq[idx], self._heapq[left])
        right_bad = right < len(self._heapq) and self._gt(self._heapq[idx], self._heapq[right])

        if left_bad and right_bad:
            if self._gt(self._heapq[left], self._heapq[right]):
                self._swap(idx, right, remap)
                return self._topdown(right, remap)
            else:
                self._swap(idx, left, remap)
                return self._topdown(left, remap)
        elif right_bad:
            self._swap(idx, right, remap)
            return self._topdown(right, remap)
        elif left_bad:
            self._swap(idx, left, remap)
            return self._topdown(left, remap)
        return idx

    def _remap_positions(self):
        self._pos = dict()
        i = 0
        while i < len(self._heapq):
            self._pos[self._heapq[i]] = i
            i+=1

    def _heapify(self):
        self._heapq = list(self._inv)
        for i in range(len(self._heapq)-1, -1, -1):
            self._topdown(i, remap=False)
        self._remap_positions()

    def _addinverse(self, k, v):
        if self._isordered():
            lst = self._inv.setdefault(v, list())
            try:
                lst.remove(k)
            except:
                pass
            lst.append(k)
        else:
            self._inv.setdefault(v, set()).add(k)

    def _additems(self, pairs):
        num_new = 0
        for k, v in pairs:
            if v not in self._inv:
                num_new += 1
            self._addinverse(k, v)
        return num_new

    @staticmethod
    def _process_args(mapping=(), **kwargs):
        if hasattr(mapping, items):
            mapping = getattr(mapping, items)()
        all_updates = list(chain(mapping, getattr(kwargs, items)()))
        real_updates = list()
        already_updated = set()
        for i in range(len(all_updates)-1, -1, -1):
            k, v = all_updates[i]
            if k not in already_updated:
                already_updated.add(k)
                real_updates.append(all_updates[i])
        real_updates.reverse()
        return real_updates

    def _preremove(self, removals):
        lost_values = set()
        for k in removals:
            v = self[k]
            self._inv[v].remove(k)
            if len(self._inv[v]) < 1: lost_values.add(v)
        return lost_values

    def _postclean(self, removes):
        count = 0
        for v in removes:
            if len(self._inv[v]) < 1:
                del self._inv[v]
                count += 1
        return count

    def _unordered_makeheap(self, newheap, mapping, **kwargs):
        if hasattr(mapping, items):
            mapping = getattr(mapping, items)()
        updates = dict(chain(mapping, getattr(kwargs, items)()))

        if not newheap: lost_values = self._preremove(set(self) & set(updates))
        num_new = self._additems(getattr(updates, items)())
        if not newheap:
            num_lost = self._postclean(lost_values)
        else:
            num_lost = 0
        if num_new > 0 or num_lost > 0:
            self._heapify()

        return updates

    def __init__(self, mapping=(), key=lambda x: x, reverse=False, tiebreak=None, __gt=None, **kwargs):
        if __gt is None:
            self._gt = lambda x, y: (key(x) > key(y)) != reverse
        else:
            self._gt = __gt
        self.tiebreak = tiebreak
        self._heapq = []
        self._pos = dict()
        self._inv = dict()
        if self.tiebreak is None:
            super(heapqup, self).__init__(self._unordered_makeheap(True, mapping, **kwargs))
        else:
            updates = self._process_args(mapping, **kwargs)
            self._additems(updates)
            self._heapify()
            super(heapqup, self).__init__(updates)

    def update(self, mapping=(), **kwargs):
        if self.tiebreak is None:
            super(heapqup, self).update(self._unordered_makeheap(False, mapping, **kwargs))
        else:
            updates = self._process_args(mapping, **kwargs)
            lost_values = self._preremove([pair[0] for pair in updates])
            num_new = self._additems(updates)
            num_lost = self._postclean(lost_values)
            if num_new > 0 or num_lost > 0:
                self._heapify()
            super(heapqup, self).update(updates)

    def _heapremove(self, v):
        """remove a priority value from the heap"""

        idx = self._pos[v]                          # get the position of the priority being removed

        if idx == len(self._heapq)-1:               # if the position is the last in the heap
            self._heapq.pop()                           # just remove it from the heap list
            del self._pos[v]                            # remove the position mapping for the priority
        elif idx == 0:                              # if the position is the first in the heap
            self._heapq[idx] = self._heapq.pop()        # move the last priority to the top
            del self._pos[v]                            # remove the position mapping for replaced priority
            self._pos[self._heapq[idx]] = idx           # update position of moved priority
            self._topdown(idx)                          # top-down heapify
        else:                                       # if the position is in the middle of the heap
            self._heapq[idx] = self._heapq.pop()        # move the last priority into position idx
            del self._pos[v]                            # remove the position mapping for replaced priority
            self._pos[self._heapq[idx]] = idx           # update position of moved priority
            parent = self._parent(idx)                  # check if the moved priority needs to be bubbled up or down
            if self._gt(self._heapq[parent], self._heapq[idx]):
                self._bottomup(idx)
            else:
                self._topdown(idx)

    def _removeitem(self, k):
        """remove a key from this data structure"""
        v = self[k]                     # get this key's priority (v)
        self._inv[v].remove(k)          # remove this key from the set of keys with priority v
        if len(self._inv[v]) < 1:       # if there are no more keys with priority v, take v out of the heap...
            del self._inv[v]                    # remove v from the priorities->keys map
            self._heapremove(v)                 # remove v from the heap list

    def _bottomup(self, idx):
        if idx > 0:
            parent = self._parent(idx)
            if self._gt(self._heapq[parent], self._heapq[idx]):
                self._swap(parent, idx)
                return self._bottomup(parent)
        return idx

    def _heapadd(self, v):
        self._heapq.append(v)
        return self._bottomup(len(self._heapq)-1)

    def _additem(self, k, v):
        if v not in self._inv:
            self._pos[v] = self._heapadd(v)
        self._addinverse(k, v)

    def __setitem__(self, k, v):
        if k in self:
            if v != self[k]:
                self._removeitem(k)
                self._additem(k, v)
            elif v == self[k] and self._isordered():
                self._addinverse(k, v)
        else:
            self._additem(k, v)

        return super(heapqup, self).__setitem__(k, v)

    def __delitem__(self, k):
        if k in self:
            self._removeitem(k)
        return super(heapqup, self).__delitem__(k)

    def copy(self):  # don't delegate w/ super - dict.copy() -> dict :(
        return type(self)(self, tiebreak=self.tiebreak, __gt=self._gt)

    @classmethod
    def fromkeys(cls, keys, v=None):
        return cls(zip(keys, [v]*len(keys)))

    def pop(self, k, v=_RaiseKeyError):
        if k in self:
            self._removeitem(k)
        if v is _RaiseKeyError:
            return super(heapqup, self).pop(k)
        return super(heapqup, self).pop(k, v)

    def popitem(self):
        if len(self._heapq) < 1:
            return super(heapqup, self).popitem()
        else:
            v = self._heapq[0]
            if self.tiebreak is self.FIFO:
                k = self._inv[v].pop(0)
            elif self.tiebreak is not None and self.tiebreak is not self.LIFO:
                k = self.tiebreak(self._inv[v])
                self._inv[v].remove(k)
            else:
                k = self._inv[v].pop()
            if len(self._inv[v]) < 1:
                del self._inv[v]
                self._heapremove(v)
            return (k, super(heapqup, self).pop(k))

    def poll(self):
        if len(self._heapq) < 1:
            raise KeyError
        else:
            return self.popitem()[0]

    def peek(self):
        if len(self._heapq) < 1:
            raise KeyError
        else:
            v = self._heapq[0]
            if self.tiebreak is self.FIFO:
                return self._inv[v][0]
            elif self.tiebreak is self.LIFO:
                return self._inv[v][-1]
            elif self.tiebreak is None:
                k = self._inv[v].pop()
                self._inv[v].add(k)
                return k
            else:
                return self.tiebreak(self._inv[v])
    
    def peek_all(self):
        if len(self._heapq) < 1:
            raise KeyError
        else:
            return self._inv[self._heapq[0]]

    def setdefault(self, k, default=None):
        if k not in self:
            self._additem(k, default)
        return super(heapqup, self).setdefault(k, default)

    def offer(self, k, default=None):
        return self.setdefault(k, default)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, super(heapqup, self).__repr__())
