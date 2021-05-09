from tools import *
class Parser:
    def __init__(self):
        self.operators2 = ['<<', '>>', '->', '//']
        self.operators1 = ['+', '-', '*', '/', '[', ']', '&', '(', ')', '.', '{', '}', ';',',','=','%','^','|']
        self.operators = self.operators1.copy()
        self.operators.extend(self.operators2)

    def is_prefix_of_operator(self, s:str):
        for op in self.operators:
            if op.startswith(s):
                return True
        return False

    def indicate_operator(self, ch:Ch):
        if self.is_prefix_of_operator(ch.char):
            return True
        return False

    def indicate_name(self, ch:Ch):
        '''
        return True if ch is beginning of a `name`
        '''
        char = ch.char
        if char.isalpha() or char=='_':
            return True

    def indicate_number(self, ch:Ch):
        '''
        @return: True if ch is beginning of a `number`
        '''
        char = ch.char
        if char.isdigit():
            return True
        return False
    
    def indicate(self, ch:Ch):
        '''
        @return: 'name',  'number' or 'operator'
        '''
        if self.indicate_name(ch):
            return 'name'
        elif self.indicate_number(ch):
            return 'number'
        elif self.indicate_operator(ch):
            return 'operator'
        else:
            logging.error(cf.red(f"CANNOT indicate type of ch = {ch.char}"))
            raise CompileException(f"CANNOT indicate type of ch = {ch.char}")

    def find_end_of_current_name(self, chs:list, index:int):
        '''
        @exmaple:
            chs='param abc{',
            index = 6 (pointer to 'a')
            return: 9 (next to last of 'abc'/ index of '{'})
        '''
        while index < len(chs):
            ch:Ch = chs[index]
            if ch.char.isalpha() or ch.char.isdigit() or ch.char == '_':
                index+=1
                continue
            return index
        return index
    def find_end_of_current_number(self, chs: list, index:int):
        while index < len(chs):
            ch: Ch = chs[index]
            if ch.char.isdigit():
                index+=1
                continue
            return index
        return index
    def find_end_of_current_operator(self, chs:list, index:int):
        prefix = ''
        while index < len(chs):
            ch: Ch = chs[index]
            prefix += ch.char
            if self.is_prefix_of_operator(prefix):
                index+=1
                continue
            return index
        return index

    def next_token(self, chs:list, index:int):
        '''
        @return: a list containing attributes of token starts from index
            'end': last index (+1) of current token
            'token': a list of Ch
            'type': one of ['name', 'number', 'operator']
            'str': string format of current token
        '''
        ans = dict()
        while index < len(chs):
            ch:Ch = chs[index]
            if ch.char.isspace():
                index+=1
                continue
            current_type = self.indicate(ch)
            if current_type == 'name':
                end = self.find_end_of_current_name(chs, index)
            elif current_type == 'number':
                end = self.find_end_of_current_number(chs, index)
            elif current_type == 'operator':
                end = self.find_end_of_current_operator(chs, index)
            token = chs[index:end]
            ans['end'] = end
            ans['token'] = token
            ans['type'] = current_type
            ans['str'] = chs_to_str(token)
            return ans
        ans['end'] = index
        ans['token'] = []
        ans['type'] = 'space'
        ans['str'] = ''
        return ans
    
    def parse_all(self, chs:list):
        index = 0
        ans = []
        while index < len(chs):
            ret = self.next_token(chs, index)
            ans.append(ret)
            index = ret['end']
        return ans
    

            

if __name__ == '__main__':
    chs = read_file(r'pse/terngrad.pse.txt')
    parser = Parser()
    ret = parser.next_token(chs, 0)
    logging.debug(ret)
    ret = parser.next_token(chs, 5)
    logging.debug(ret)
    index = 0
    while index < len(chs):
        ret = parser.next_token(chs, index)
        logging.debug(ret['str'])
        index = ret['end']
    