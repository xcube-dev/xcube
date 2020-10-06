import unittest

from xcube.core.store.typeid import TypeId

class TypeIdTest(unittest.TestCase):

    def test_name(self):
        type_id = TypeId(name='tfsr')
        self.assertEqual('tfsr', type_id.name)
        self.assertEqual(set(), type_id.flags)

    def test_name_and_flags(self):
        type_id = TypeId(name='tfsr', flags={'fzht', 'gdfhg'})
        self.assertEqual('tfsr', type_id.name)
        self.assertEqual({'fzht', 'gdfhg'}, type_id.flags)

    def test_string(self):
        type_id = TypeId(name='tfsr', flags={'fzht', 'gdfhg'})
        self.assertEqual(str(type_id), 'tfsr[fzht,gdfhg]')

    def test_equality(self):
        type_id = TypeId(name='tfsr', flags={'fzht'})
        self.assertFalse(type_id == TypeId('dtftg'))
        self.assertFalse(type_id == TypeId('dtftg', flags={'fzht'}))
        self.assertFalse(type_id == TypeId('tfsr'))
        self.assertFalse(type_id == TypeId('tfsr', flags={'dsf'}))
        self.assertFalse(type_id == TypeId('tfsr', flags={'dsf', 'fzht'}))
        self.assertTrue(type_id == TypeId('tfsr', flags={'fzht'}))

    def test_equality_flag_order_is_irrelevant(self):
        type_id_1 = TypeId(name='tfsr', flags={'fzht', 'duxdrh5g'})
        type_id_2 = TypeId(name='tfsr', flags={'duxdrh5g', 'fzht'})
        self.assertTrue(type_id_1  == type_id_2)

    def test_is_compatible(self):
        type_id = TypeId(name='tfsr', flags={'fzht'})
        self.assertFalse(type_id.is_compatible(TypeId('dtftg')))
        self.assertFalse(type_id.is_compatible(TypeId('dtftg', flags={'fzht'})))
        self.assertFalse(type_id.is_compatible(TypeId('tfsr')))
        self.assertFalse(type_id.is_compatible(TypeId('tfsr', flags={'dsf'})))
        self.assertTrue(type_id.is_compatible(TypeId('tfsr', flags={'dsf', 'fzht'})))
        self.assertTrue(type_id.is_compatible(TypeId('tfsr', flags={'fzht'})))

    def test_normalize(self):
        tfsr_flagged_id = TypeId(name='tfsr', flags={'fzht'})
        self.assertEqual(TypeId.normalize(tfsr_flagged_id), tfsr_flagged_id)
        self.assertEqual(TypeId.normalize('tfsr[fzht]'), tfsr_flagged_id)

        tfsr_id = TypeId(name='tfsr')
        self.assertEqual(TypeId.normalize(tfsr_id), tfsr_id)
        self.assertEqual(TypeId.normalize('tfsr'), tfsr_id)

    def test_parse(self):
        # tfsr_flagged_id = TypeId(name='tfsr', flags={'fzht'})
        parsed_id = TypeId.parse('tfsr[fzht,rer]')
        self.assertEqual('tfsr', parsed_id.name)
        self.assertEqual({'fzht', 'rer'}, parsed_id.flags)

    def test_parse_exception(self):
        try:
            TypeId.parse('An unparseable expression[')
            self.fail('This should have failed')
        except SyntaxError as se:
            self.assertEqual(se.msg, '"An unparseable expression[" cannot be parsed: No end brackets found')
