from collections import defaultdict
from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

from skidl.pin import pin_types

SKIDL_lib_version = '0.0.1'

net_var = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'DS8830', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'DS8830'}), 'ref_prefix':'U', 'fplist':['Package_DIP:DIP-14_W7.62mm', 'Package_DIP:DIP-14_W7.62mm'], 'footprint':'Package_DIP:DIP-14_W7.62mm', 'keywords':'Dual differential line driver', 'description':'Dual differential line driver and dual four-input NAND or dual four-input AND function, VDD +5V, DIP-14', 'datasheet':'http://pdf1.alldatasheet.com/datasheet-pdf/view/8473/NSC/DS7830J.html', 'pins':[
            Pin(num='1',name='A_1',func=pin_types.INPUT,unit=1),
            Pin(num='2',name='A_2',func=pin_types.INPUT,unit=1),
            Pin(num='3',name='A_3',func=pin_types.INPUT,unit=1),
            Pin(num='4',name='A_4',func=pin_types.INPUT,unit=1),
            Pin(num='5',name='A_AND_OUTPUT',func=pin_types.INPUT,unit=1),
            Pin(num='6',name='A_NAND_OUTPUT',func=pin_types.INPUT,unit=1),
            Pin(num='7',name='GND',func=pin_types.PWRIN,unit=1),
            Pin(num='8',name='B_NAND_OUTPUT',func=pin_types.INPUT,unit=1),
            Pin(num='9',name='B_AND_OUTPUT',func=pin_types.INPUT,unit=1),
            Pin(num='10',name='B_4',func=pin_types.INPUT,unit=1),
            Pin(num='11',name='B_3',func=pin_types.INPUT,unit=1),
            Pin(num='12',name='B_2',func=pin_types.INPUT,unit=1),
            Pin(num='13',name='B_1',func=pin_types.INPUT,unit=1),
            Pin(num='14',name='VCC',func=pin_types.PWRIN,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'Conn_01x12', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_01x12'}), 'ref_prefix':'J', 'fplist':[''], 'footprint':'Connector_PinSocket_2.54mm:PinSocket_1x12_P2.54mm_Vertical', 'keywords':'connector', 'description':'Generic connector, single row, 01x12, script generated (kicad-library-utils/schlib/autogen/connector/)', 'datasheet':'', 'pins':[
            Pin(num='1',name='Pin_1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='Pin_2',func=pin_types.PASSIVE,unit=1),
            Pin(num='3',name='Pin_3',func=pin_types.PASSIVE,unit=1),
            Pin(num='4',name='Pin_4',func=pin_types.PASSIVE,unit=1),
            Pin(num='5',name='Pin_5',func=pin_types.PASSIVE,unit=1),
            Pin(num='6',name='Pin_6',func=pin_types.PASSIVE,unit=1),
            Pin(num='7',name='Pin_7',func=pin_types.PASSIVE,unit=1),
            Pin(num='8',name='Pin_8',func=pin_types.PASSIVE,unit=1),
            Pin(num='9',name='Pin_9',func=pin_types.PASSIVE,unit=1),
            Pin(num='10',name='Pin_10',func=pin_types.PASSIVE,unit=1),
            Pin(num='11',name='Pin_11',func=pin_types.PASSIVE,unit=1),
            Pin(num='12',name='Pin_12',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] })])