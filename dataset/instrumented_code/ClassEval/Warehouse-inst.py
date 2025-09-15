import inspect
import json
import os
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def recursive_object_seralizer(obj, visited):
    seralized_dict = {}
    keys = list(obj.__dict__)
    for k in keys:
        if id(obj.__dict__[k]) in visited:
            seralized_dict[k] = "<RECURSIVE {}>".format(obj.__dict__[k])
            continue
        if isinstance(obj.__dict__[k], (float, int, str, bool, type(None))):
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], tuple):
            ## handle tuple
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], set):
            ## handle set
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], list):
            ## handle list
            seralized_dict[k] = obj.__dict__[k]
        elif hasattr(obj.__dict__[k], '__dict__'):
            ## handle object
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], dict):
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif callable(obj.__dict__[k]):
            ## handle function
            if hasattr(obj.__dict__[k], '__name__'):
                seralized_dict[k] = "<function {}>".format(obj.__dict__[k].__name__)
        else:
            seralized_dict[k] = str(obj.__dict__[k])
    return seralized_dict

def inspect_code(func):
   def wrapper(*args, **kwargs):
       visited = []
       json_base = "/home/changshu/ClassEval/data/benchmark_solution_code/input-output/"
       if not os.path.exists(json_base):
           os.mkdir(json_base)
       jsonl_path = json_base + "/Warehouse.jsonl"
       para_dict = {"name": func.__name__}
       args_names = inspect.getfullargspec(func).args
       if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
           ## 'self'
           self_args = args[0]
           para_dict['self'] = recursive_object_seralizer(self_args, [id(self_args)])
       else:
           para_dict['self'] = {}
       if len(args) > 0 :
           if args_names[0] == 'self':
               other_args = {}
               for m,n in zip(args_names[1:], args[1:]):
                   other_args[m] = n
           else:
               other_args = {}
               for m,n in zip(args_names, args):
                   other_args[m] = n
           
           para_dict['args'] = other_args
       else:
           para_dict['args'] = {}
       if kwargs:
           para_dict['kwargs'] = kwargs
       else:
           para_dict['kwargs'] = {}
          
       result = func(*args, **kwargs)
       para_dict["return"] = result
       with open(jsonl_path, 'a') as f:
           f.write(json.dumps(para_dict, default=custom_serializer) + "\n")
       return result
   return wrapper


'''
# The class manages inventory and orders, including adding products, updating product quantities, retrieving product quantities, creating orders, changing order statuses, and tracking orders.

class Warehouse:
    def __init__(self):
        """
        Initialize two fields.
        self.inventory is a dict that stores the products.
        self.inventory = {Product ID: Product}
        self.orders is a dict that stores the products in a order.
        self.orders = {Order ID: Order}
        """
        self.inventory = {}  # Product ID: Product
        self.orders = {}  # Order ID: Order

    def add_product(self, product_id, name, quantity):
        """
        Add product to inventory and plus the quantity if it has existed in inventory.
        Or just add new product to dict otherwise.
        :param product_id: int
        :param name: str, product name
        :param quantity: int, product quantity
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.inventory
        {1: {'name': 'product1', 'quantity': 3}}
        """

    def update_product_quantity(self, product_id, quantity):
        """
        According to product_id, add the quantity to the corresponding product in inventory.
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.update_product_quantity(1, -1)
        >>> warehouse.inventory
        {1: {'name': 'product1', 'quantity': 2}}
        """

    def get_product_quantity(self, product_id):
        """
        Get the quantity of specific product by product_id.
        :param product_id, int
        :return: if the product_id is in inventory then return the corresponding quantity,
                or False otherwise.
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.get_product_quantity(1)
        3
        >>> warehouse.get_product_quantity(2)
        False
        """

    def create_order(self, order_id, product_id, quantity):
        """
        Create a order which includes the infomation of product, like id and quantity.
        And put the new order into self.orders.
        The default value of status is 'Shipped'.
        :param order_id: int
        :param product_id: int
        :param quantity: the quantity of product that be selected.
        :return False: only if product_id is not in inventory or the quantity is not adequate
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.create_order(1, 1, 2)
        >>> warehouse.orders
        {1: {'product_id': 1, 'quantity': 2, 'status': 'Shipped'}}
        >>> warehouse.create_order(1, 2, 2)
        False
        """

    def change_order_status(self, order_id, status):
        """
        Change the status of order if the input order_id is in self.orders.
        :param order_id: int
        :param status: str, the state that is going to change to
        :return False: only if the order_id is not in self.orders
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.create_order(1, 1, 2)
        >>> warehouse.change_order_status(1, "done")
        >>> warehouse.orders
        {1: {'product_id': 1, 'quantity': 2, 'status': 'done'}}
        """

    def track_order(self, order_id):
        """
        Get the status of specific order.
        :param order_id: int
        :return False: only if the order_id is not in self.orders.
        >>> warehouse.add_product(1, "product1", 3)
        >>> warehouse.create_order(1, 1, 2)
        >>> warehouse.track_order(1)
        'Shipped'
        """
'''

class Warehouse:
    def __init__(self):
        self.inventory = {}  # Product ID: Product
        self.orders = {}  # Order ID: Order

    @inspect_code
    def add_product(self, product_id, name, quantity):
        if product_id not in self.inventory:
            self.inventory[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.inventory[product_id]['quantity'] += quantity

    @inspect_code
    def update_product_quantity(self, product_id, quantity):
        if product_id in self.inventory:
            self.inventory[product_id]['quantity'] += quantity

    @inspect_code
    def get_product_quantity(self, product_id):
        if product_id in self.inventory:
            return self.inventory[product_id]['quantity']
        else:
            return False

    @inspect_code
    def create_order(self, order_id, product_id, quantity):
        if self.get_product_quantity(product_id) >= quantity:
            self.update_product_quantity(product_id, -quantity)
            self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Shipped'}
        else:
            return False

    @inspect_code
    def change_order_status(self, order_id, status):
        if order_id in self.orders:
            self.orders[order_id]['status'] = status
        else:
            return False

    @inspect_code
    def track_order(self, order_id):
        if order_id in self.orders:
            return self.orders[order_id]['status']
        else:
            return False

import unittest


class WarehouseTestAddProduct(unittest.TestCase):
    def test_add_product_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 1', 'quantity': 10}})

    def test_add_product_2(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.add_product(2, 'product 2', 5)
        self.assertEqual(warehouse.inventory,
                         {1: {'name': 'product 1', 'quantity': 10}, 2: {'name': 'product 2', 'quantity': 5}})

    def test_add_product_3(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 3', 10)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 3', 'quantity': 10}})

    def test_add_product_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 4', 10)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 4', 'quantity': 10}})

    def test_add_product_5(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 5', 10)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 5', 'quantity': 10}})

    def test_add_product_6(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 5', 10)
        warehouse.add_product(1, 'product 5', 10)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 5', 'quantity': 20}})


class WarehouseTestUpdateProductQuantity(unittest.TestCase):
    def test_update_product_quantity_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.update_product_quantity(1, 5)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 1', 'quantity': 15}})

    # quantity is negative
    def test_update_product_quantity_2(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.update_product_quantity(1, -5)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 1', 'quantity': 5}})

    def test_update_product_quantity_3(self):
        warehouse = Warehouse()
        warehouse.update_product_quantity(1, -5)
        self.assertEqual(warehouse.inventory, {})

    def test_update_product_quantity_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.update_product_quantity(1, 1)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 1', 'quantity': 11}})

    def test_update_product_quantity_5(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.update_product_quantity(1, -9)
        self.assertEqual(warehouse.inventory, {1: {'name': 'product 1', 'quantity': 1}})


class WarehouseTestGetProductQuantity(unittest.TestCase):
    def test_get_product_quantity_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        self.assertEqual(warehouse.get_product_quantity(1), 10)

    def test_get_product_quantity_2(self):
        warehouse = Warehouse()
        self.assertEqual(warehouse.get_product_quantity(1), False)

    def test_get_product_quantity_3(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 5)
        self.assertEqual(warehouse.get_product_quantity(1), 5)

    def test_get_product_quantity_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 100)
        self.assertEqual(warehouse.get_product_quantity(1), 100)

    def test_get_product_quantity_5(self):
        warehouse = Warehouse()
        warehouse.add_product(5, 'product 1', 10)
        self.assertEqual(warehouse.get_product_quantity(5), 10)


class WarehouseTestCreateOrder(unittest.TestCase):
    def test_create_order_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.create_order(1, 1, 5)
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 5, 'status': 'Shipped'}})

    def test_create_order_2(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        result = warehouse.create_order(1, 1, 15)
        self.assertFalse(result)

    def test_create_order_3(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 1)
        warehouse.create_order(1, 1, 1)
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 1, 'status': 'Shipped'}})

    def test_create_order_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 4', 5)
        warehouse.create_order(1, 1, 5)
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 5, 'status': 'Shipped'}})

    def test_create_order_5(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 5', 100)
        warehouse.create_order(1, 1, 50)
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 50, 'status': 'Shipped'}})


class WarehouseTestChangeOrderStatus(unittest.TestCase):
    def test_change_order_status_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.create_order(1, 1, 5)
        warehouse.change_order_status(1, 'Delivered')
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 5, 'status': 'Delivered'}})

    def test_change_order_status_2(self):
        warehouse = Warehouse()
        result = warehouse.change_order_status(1, 'Delivered')
        self.assertFalse(result)

    def test_change_order_status_3(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 3', 5)
        warehouse.create_order(1, 1, 5)
        warehouse.change_order_status(1, 'Delivered')
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 5, 'status': 'Delivered'}})

    def test_change_order_status_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 4', 100)
        warehouse.create_order(1, 1, 50)
        warehouse.change_order_status(1, 'Delivered')
        self.assertEqual(warehouse.orders, {1: {'product_id': 1, 'quantity': 50, 'status': 'Delivered'}})

    def test_change_order_status_5(self):
        warehouse = Warehouse()
        result = warehouse.change_order_status(2, 'Delivered')
        self.assertFalse(result)


class WarehouseTestTrackOrder(unittest.TestCase):
    def test_track_order_1(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        warehouse.create_order(1, 1, 5)
        self.assertEqual(warehouse.track_order(1), 'Shipped')

    def test_track_order_2(self):
        warehouse = Warehouse()
        result = warehouse.track_order(1)
        self.assertFalse(result)

    def test_track_order_3(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 3', 10)
        warehouse.create_order(1, 1, 1)
        self.assertEqual(warehouse.track_order(1), 'Shipped')

    def test_track_order_4(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 4', 100)
        warehouse.create_order(1, 1, 50)
        self.assertEqual(warehouse.track_order(1), 'Shipped')

    def test_track_order_5(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 5', 100)
        warehouse.create_order(1, 1, 10)
        self.assertEqual(warehouse.track_order(1), 'Shipped')


class WarehouseTestMain(unittest.TestCase):
    def test_main(self):
        warehouse = Warehouse()
        warehouse.add_product(1, 'product 1', 10)
        self.assertEqual({1: {'name': 'product 1', 'quantity': 10}}, warehouse.inventory)

        warehouse.update_product_quantity(1, -5)
        self.assertEqual({1: {'name': 'product 1', 'quantity': 5}}, warehouse.inventory)

        self.assertEqual(warehouse.get_product_quantity(1), 5)

        warehouse.create_order(1, 1, 3)
        self.assertEqual({1: {'product_id': 1, 'quantity': 3, 'status': 'Shipped'}}, warehouse.orders)

        warehouse.change_order_status(1, 'Delivered')
        self.assertEqual({1: {'product_id': 1, 'quantity': 3, 'status': 'Delivered'}}, warehouse.orders)

        self.assertEqual('Delivered', warehouse.track_order(1))

