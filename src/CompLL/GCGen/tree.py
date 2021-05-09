import copy
from parser import Parser

from tools import *
from unit import *


class Tree:
    def __init__(self, filename:str):
        self.filename = filename
        self.chs = read_file(filename)
        self.parser = Parser()
        self.tokens = self.parser.parse_all(self.chs)
        self.i = 0 # index of self.tokens
        self.params = dict()
        self.funcs = dict()
        self.context = dict()
        self.contexts = [self.context]

    def record_context(f):
        def wrapper(self, *args, **kwargs):
            # context = copy.deepcopy(self.contexts[-1])
            context = copy.copy(self.contexts[-1])
            self.contexts.append(context)
            self.context = self.contexts[-1]
            ret = f(self, *args, **kwargs)
            self.contexts = self.contexts[:-1]
            self.context = self.contexts[-1]
            return ret
        return wrapper


    def generate_bug(self, msg:str):
        assert(self.i>0)
        t = self.tokens[self.i-1]
        token = t['token']
        ch:Ch = token[0]
        bug_msg = cf.red(f"[FILENAME:{self.filename}][LINE:{ch.line_no},{ch.column_no}][MSG:{msg}]")
        return bug_msg


    def next_token(self):
        token = self.tokens[self.i]
        self.i+=1
        return token

    @record_context
    def block_typename(self):
        '''
        @return: Utype
        '''
        ans = dict()
        basic_type = self.next_token()
        ans['typename'] = basic_type['str']
        ans["is_vector"] = False
        ans["bitwidth"] = ''
        t = self.next_token()
        if t['str'] == '<':
            # elongated bitwidth
            bitwidth = self.block_right_expression(l_var=None,end='>')
            is_vector = False
            tt = self.next_token()
            if tt['str'] == '*':
                is_vector = True
            else:
                self.i-=1
            logging.debug(cf.yellow(f"bitwidth={bitwidth}"))
            utype = Utype(ans['typename'], bitwidth['str'], is_vector, self.context)
            return utype
            # raise CompileException("NOT implement now")
        else:
            self.i -= 1 # roll index back
        t = self.next_token()
        if t['str'] == '*':
            ans["is_vector"] = True
        else:
            self.i -= 1
        utype = Utype(ans['typename'], ans['bitwidth'], ans['is_vector'], self.context)
        return utype

    @record_context
    def block_one_param(self):
        '''
        @explanation: to resolve such patterns: "int a", "uint<bitwidth> abc"
        @return: an instance of Var
        '''
        utype = self.block_typename()
        p_name = self.next_token()
        p_var = Variable(p_name['str'], utype)
        logging.debug(f"p_var={p_var}")
        return p_var

    @record_context
    def block_param_define(self):
        assert(self.tokens[self.i-1]['str'] == 'param')
        name = self.next_token()
        assert(name['type']=='name') # not operator nor number
        param_name = name['str']
        logging.debug(f'param_name={param_name}')
        param = Param(param_name)
        token = self.next_token()
        assert(token['str'] == '{')
        while True:
            t = self.next_token()
            if t['str'] == '}':
                break
            self.i-=1
            one_param = self.block_one_param()
            param.add_var(one_param)
            t = self.next_token()
            if t['str'] != ';':
                raise CompileException(self.generate_bug(f"Behind definitioin of param, ';' is expected!"))         
        return param

    def fetch_and_check(self, expected_str):
        t = self.next_token()
        if t['str'] != expected_str:
            raise CompileException(f"Expected {expected_str}")
        
    def block_extract(self, l_var:None):
        if l_var != None:
            raise CompileException("Do not support left value for operator `extract`")
        self.fetch_and_check("extract")
        self.fetch_and_check('(')
        current_unit:Func = self.context['current_unit']
        params = []
        while True:
            t = self.next_token()
            if t['str'] not in self.context:
                raise CompileException("Unknown parameter")
            v:Variable = self.context[t['str']]
            params.append(v)
            t = self.next_token()
            if t['str'] == ',':
                continue
            elif t['str'] == ')':
                break
        offset = 0
        u:Variable = params[0]
        if u.utype.is_vector == False:
            raise CompileException("Variable to extract is not a vector")
        s = ""
        for i in range(1,len(params)):
            v:Variable = params[i]
            if v.utype.is_vector == False:
                s += f"get_policy<policy_t>::memcpyOut(&{v.name}, {u.name}+{offset}, {v.bytes}, stream);\n"
            else:
                if i < len(params)-1: # not last 
                    raise CompileException("Please only use vector as last parameter in `extract`")
                v.set_initialize_code(f"{v.utype.generate_call()} {v.name} = reinterpret_cast<{v.utype.generate_call()}>({u.name}+{offset})")
                v.set_bits((u.bytes-offset)*8)
            offset += v.bytes
        ans = dict()
        ans['str'] = s
        ans['return_type'] = "none"
        return ans



            


    @record_context
    def block_concat(self, l_var:Variable):
        s = ""
        ans= dict()
        ans['return_type'] = 'todo'
        current_unit:Func = self.context['current_unit']
        if l_var.name not in current_unit.params:
            raise CompileException("left value of `concat` must be params of Func")
        logging.debug(f"l_var={l_var}")
        self.fetch_and_check('concat')
        self.fetch_and_check('(')
        offset = 0
        while True:
            t = self.next_token()
            ele:Variable = self.context[t['str']]
            if ele.utype.is_vector == False:
                s += f"get_policy<policy_t>::memcpyIn({l_var.name}+({offset}), &{ele.name}, {ele.bytes}, stream);\n"
            else:
                logging.debug(cf.red(f"ele={ele}"))
                ele.set_initialize_code(f"{ele.utype.generate_call()} {ele.name}={l_var.name}+({offset})")
                logging.debug(cf.red(f"ele={ele}"))
            offset += ele.bytes
            t = self.next_token()
            if t['str'] == ')':
                break
            elif t['str'] == ',':
                pass
            else:
                raise CompileException("Expected ',' or ')' here")
        ans['str'] = s
        return ans


    @record_context
    def block_map(self, l_var:Variable):
        ans = dict()
        self.fetch_and_check('map')
        self.fetch_and_check('(')
        t = self.next_token()
        input_vector = None
        input_len = 0
        output_bits = 0
        if t['str'] == 'range':
            self.fetch_and_check('(')
            ret = self.block_right_expression(l_var=None, end=')')
            input_len = sympy.Symbol(ret['str'])
            param1 = "thrust::counting_iterator<int32_t>(0)"
            # param2 = f"thrust::counting_iterator<int32_t>({ret['str']})"
        else:
            # make sure it is vector
            # assign input_vector
            # assign input_len
            pass
        self.fetch_and_check(',')
        t = self.next_token()
        if t['str'] in self.context:
            lambda_func:Lambda_func = self.context[t['str']]
            if type(lambda_func) != Lambda_func:
                raise CompileException(f"{t['str']} expected to be a Lambda_func")
            output_bits = input_len * lambda_func.return_type.bitwidth
            if lambda_func.need_aggregation():
                logging.info(f"input_len={input_len}")
                logging.info(f"lambda_func.aggregate_rate()={lambda_func.aggregate_rate()}")
                aggregated_len = (input_len + lambda_func.aggregate_rate() - 1)//lambda_func.aggregate_rate()
            else:
                aggregated_len = input_len
            param2 = f"{param1} + ({aggregated_len})"
            param4 = lambda_func.generate_call(input_len)
        else:
            raise CompileException(f"Unrecognized {t}")
        self.fetch_and_check(')')
        self.fetch_and_check(';')
        if l_var != None:
            param3 = l_var.name
            l_var.set_bits(output_bits)
            ans['str'] = f'thrust::transform(policy,{param1},{param2},{param3},{param4});'
            # ans['str'] = f'thrust::for_each(policy,{param1},{param2},{param3},{param4});'
            ans['return_type'] = 'todo'
        else:
            # thrust::for_each
            pass
        return ans


    @record_context
    def block_reduce(self, l_var=None):
        '''
        @return:
            str:    translated expression
            return_type
        '''
        self.fetch_and_check('reduce')
        self.fetch_and_check('(')
        vector = self.next_token()
        logging.debug(cf.red(f"self.context={self.context}"))
        if vector['str'] not in self.context:
            raise CompileException(f"Unrecognized varname {vector['str']}")
        vector:Variable = self.context[vector['str']]
        if vector.utype.is_vector == False:
            raise CompileException(f"Expected vector type")
        param1 = vector.name
        self.fetch_and_check(',')
        ret = self.block_right_expression(None,end=',')
        param2 = ret['str']
        # no ',', because already resolved in block_right_expression
        param3 = self.next_token()
        param3 = param3['str']
        if param3 in self.context:
            v:Lambda_func = self.context[param3]
            assert(type(v) == Lambda_func)
            param3 = v.generate_call(maximum_index=None)
        elif param3 == 'smaller':
            param3 = f"thrust::smaller<{vector.utype.name}>()"
        '''
        do something with param3
        '''
        self.fetch_and_check(')')
        logging.debug(f"param1={param1}\tparam2={param2}\tparam3={param3}")
        expr = f'thrust::reduce(policy, {vector.name}, {vector.name}+{vector.size}, {param2}, {param3})'
        ans=dict()
        if l_var == None:
            ans['str'] = expr
        else:
            ans['str'] = f"{l_var.name} = {expr}"
        ans['return_type'] = vector.utype.name
        return ans

    @record_context
    def block_lambda_func(self, l_var:Lambda_func):
        l_var:Lambda_func = l_var
        self.fetch_and_check('[')
        self.fetch_and_check('&')
        self.fetch_and_check(']')
        self.fetch_and_check('(')
        while True:
            one_param = self.block_one_param()
            l_var.add_param(one_param)
            t = self.next_token()
            if t['str'] == ')':
                break
            elif t['str'] == ',':
                continue
            else:
                raise CompileException("Expected ')' or ',' here")
        self.fetch_and_check('->')
        return_type = self.block_typename()
        l_var.set_return_type(return_type)
        if l_var.need_aggregation():
            bw = str(l_var.return_type.bitwidth)
            if bw not in self.context:
                raise CompileException(f"Invalid bitwidth: {bw}")
            bw = self.context[bw]
            l_var.add_ref(bw)
            
        logging.debug(f"block_lambda_func:\treturn_type={return_type}")
        self.fetch_and_check('{')
        # how to pick out refered vars?
        logging.debug(f"self.context={self.context}")
        end_lambda_func = self.get_index_of_right_bracket(self.i-1)
        current_unit:Func = self.context['current_unit']
        for i in range(self.i, end_lambda_func):
            t = self.tokens[i]
            s = t['str']
            if s in current_unit.vars:
                v = current_unit.vars[s]
                l_var.add_ref(v)
                if type(v) == Variable:
                    v:Variable = v
                    if v.utype.is_vector and v.utype.is_partial():
                        bitwidth = v.utype.bitwidth
                        bitwidth = str(bitwidth)
                        bitwidth = self.context[bitwidth]
                        l_var.add_ref(bitwidth)

        self.context['current_unit'] = l_var
        while True:
            t = self.next_token()
            if t['str'] == ';':
                continue
            elif t['str'] == '}':
                break
            self.i-=1
            ret = self.block_statement()
            l_var.add_statement(ret['str'])
            if 'defined_var' in ret:
                defined_var = ret['defined_var']
                l_var.add_var(defined_var)
                self.context[defined_var.name] = defined_var
        logging.debug(f"ret={ret}")
        ans = dict()
        ans['str'] = ''
        return ans
        # logging.debug(f"lambda_func={l_var}")

    def get_index_of_right_bracket(self, index_of_left_bracket:int):
        '''
        @return: if not exists, return -1
        '''
        i = index_of_left_bracket
        l = self.tokens[i]
        l = l['str']
        d = {'(':')','[':']','{':'}','[':']'}
        assert(l in d)
        r = d[l]
        depth = 0
        while i < len(self.tokens):
            t = self.tokens[i]
            if t['str'] == l:
                depth += 1
            elif t['str'] == r:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    @record_context
    def block_if(self, l_var=None):
        self.fetch_and_check('if')
        self.fetch_and_check('(')
        judge = self.block_right_expression(l_var=None, end=')')
        t = self.next_token()
        if t['str'] == '{':
            self.i -= 1
            body_if = self.block_block()
            body_if = ";\n".join(body_if)
        else:
            self.i -= 1
            body_if = self.block_statement()
            body_if = body_if['str']
        t = self.next_token()
        body_else = ""
        if t['str'] == 'else':
            t = self.next_token()
            if t['str'] == '{':
                self.i -= 1
                body_else = self.block_block()
                body_else = ";\n".join(body_else)
            else:
                self.i -= 1
                body_else = self.block_statment()
                body_else = body_else['str']
        else:
            self.i-=1
        ans = f'''if ({judge['str']}){{
            {body_if};
}}'''
        if body_else != "":
            ans += f'''else{{
            {body_else};
}}'''
        ans += ";"
        ans = {"str":ans, "return_type":"none"}
        return ans

    @record_context
    def block_right_expression(self, l_var=None, end=';'):
        '''
        it will resolve end
        @return:
            str:    translated expression
            return_type:    
        '''
        if type(l_var) == Lambda_func:
            return self.block_lambda_func(l_var)
        ans = dict()
        t = self.next_token()
        self.i-=1
        logging.debug(f"block_right_expression: t={t}") 
        if t['str'] == 'reduce':
            ret = self.block_reduce(l_var)
            self.fetch_and_check(';')
            return ret
        elif t['str'] == 'map':
            ret = self.block_map(l_var)
            return ret
        elif t['str'] == 'extract':
            ret = self.block_extract(l_var)
            return ret
        elif t['str'] == 'concat':
            ret = self.block_concat(l_var)
            return ret
        else:
            translated_tokens = []
            while True:
                t = self.next_token()
                if t['str'] == end:
                    break
                n = self.next_token()
                self.i-=1
                if n['str'] == '.':
                    p1 = t['str']
                    if p1 not in self.context:
                        raise CompileException("Unrecognized token")
                    v:Variable = self.context[p1]
                    if v.utype.is_vector:
                        self.fetch_and_check('.')
                        p2 = self.next_token()
                        p2 = p2['str']
                        if p2 == 'size':
                            translated_tokens.append(f"{v.size}")
                        else:
                            raise CompileException("Unsupported sub call for vector")
                    else:
                        raise CompileException(f"Do not support sub call for {v.name}")
                elif n['str'] == '[':
                    p1 = t['str']
                    if p1 not in self.context:
                        raise CompileException("Unrecognized token")
                    v:Variable = self.context[p1]
                    if v.utype.is_vector == False:
                        raise CompileException("Only support subscript for vector")
                    self.fetch_and_check('[')
                    index = self.block_right_expression(l_var=None,end=']')
                    if v.utype.is_partial():
                        index = f"({index['str']})"
                        index = sympy.Symbol(index)
                        bitwidth = v.utype.bitwidth
                        data_per_byte = 8 / bitwidth
                        offset = index % data_per_byte
                        index = index / data_per_byte
                        index = f"static_cast<int>({index})"
                        # mask = (1 << bitwidth) - 1
                        mask = f"(1<<{bitwidth})-1"
                        q = f"(({v.name}[{index}]>>({offset}*{bitwidth}))&({mask}))"
                        translated_tokens.append(q)
                    else:
                        index = index['str']
                        q = f"{v.name}[{index}]"
                        translated_tokens.append(q)
                elif t['str'] == 'random':
                    self.fetch_and_check('<')
                    utype:Utype = self.block_typename()
                    self.fetch_and_check('>')
                    current_func:Func = self.context['current_unit']
                    current_func.set_use_random(utype)
                    translated_tokens.append(f"random_{utype.name}")
                else:
                    translated_tokens.append(t['str'])
            translated_expr = ''.join(translated_tokens)
            if l_var != None:
                translated_expr = l_var.name + '=' + translated_expr
            ans['str'] = translated_expr
            ans['return_type'] = 'todo'
        return ans

    @record_context
    def block_statement(self):
        '''
        @return:
            str: how this sentence is translated
            defined_var
        '''
        ans = dict()
        begin_index = self.i
        t = self.next_token()
        if is_typename(t['str']):
            self.i -= 1
            one_var:Varibale = self.block_one_param()
            logging.debug(f"one_var={one_var}")
            if one_var.utype.name == 'lambda_func':
                one_var = Lambda_func(one_var.name)
            ans['defined_var'] = one_var
            t = self.next_token()
            if t['str'] == '=':
                ret = self.block_right_expression(l_var=one_var)
                ans['str'] = ret['str']
            elif t['str'] == ';':
                ans['str'] = ""
                pass
            else:
                raise CompileException(f"UNEXPECTED char: {t['str']}")
        elif t['str'] == 'if':
            self.i-=1
            ret = self.block_if(l_var=None)
            ans['str'] = ret['str']
        elif t['str'] == 'return':
            ret = self.block_right_expression(l_var=None)
            current_func:Func = self.context['current_unit']
            if current_func.need_aggregation():
                ans['str'] = f"_q = (_q << ({current_func.return_type.bitwidth})) + {ret['str']}"
            else:
                ans['str'] = f"return {ret['str']}"
        elif t['str'] in self.context:
            ele = self.context[t['str']]
            if type(ele) != Variable:
                raise CompileException(f"Variable type expected here.")
            t = self.next_token()
            if t['str'] == '=':
                ret = self.block_right_expression(ele)
                ans['str'] = ret['str']
            else:
                raise CompileException(f"UNEXPECTED char: {t['str']}")
        else: # pure call, example: sort(G, greater)
            self.i -= 1
            ret = self.block_right_expression()
            ans['str'] = ret['str']
        return ans

    @record_context
    def block_block(self):
        '''
        @return: a list of statements
        '''
        ans = []
        self.fetch_and_check('{')
        func:Func = self.context['current_unit']
        while True:
            t = self.next_token()
            if t['str'] == ';':
                continue
            elif t['str'] == '}':
                break
            self.i -= 1
            ret = self.block_statement()
            # func.add_statement(ret['str'])
            ans.append(ret['str'])
            if 'defined_var' in ret:
                defined_var = ret['defined_var']
                func.add_var(defined_var)
                self.context[defined_var.name] = defined_var
        return ans
        


    @record_context
    def block_func(self):
        return_type = self.block_typename()
        name_token = self.next_token()
        assert(name_token['type']=='name')
        func = Func(name_token['str'])
        func.set_return_type(return_type)
        self.context['current_unit'] = func
        self.context[func.name] = func
        # resolve parameters
        t = self.next_token()
        if t['str'] != '(':
            raise CompileException("'(' is expected after function name")
        while True:
            t = self.next_token()
            if t['str'] == ')':
                break
            self.i-=1
            one_param = self.block_one_param()
            self.context[one_param.name] = one_param
            func.add_param(one_param)
            t = self.next_token()
            if t['str'] == ')':
                self.i-=1
            elif t['str'] == ',':
                pass
            else:
                raise CompileException(self.generate_bug(f"',' or ')' expected after each parameter in function definition"))
        # resolve body
        t = self.next_token()
        if t['str'] != '{':
            raise CompileException("'{' is expected")
        while True:
            t = self.next_token()
            if t['str'] == ';':
                continue
            if t['str'] == '}':
                break
            logging.debug(cf.red(f"prefetch: t={t}"))
            self.i-=1
            ret = self.block_statement()
            func.add_statement(ret['str'])
            if 'defined_var' in ret:
                defined_var = ret['defined_var']
                func.add_var(defined_var)
                self.context[defined_var.name] = defined_var
        logging.info(f"func={func}")
        # func.dump_body()
        # func.dump_wrapper()
        # func.dump_register()
        return func


    def block_code(self):
        '''
        @param i: index of self.tokens
        '''
        try:
            # while self.i < len(self.tokens):
            for cnt in range(1):
                token = self.next_token()
                if token['str'] == 'param':
                    param = self.block_param_define()
                    self.context[param.name] = param
                    logging.debug(f"param = {param}")
                elif is_typename(token['str']):
                    # definition of function
                    # why not definition of variable? no need here. 
                    self.i -= 1
                    self.block_func()
        except CompileException as e:
            logging.error(self.generate_bug(e))
            raise Exception
        


if __name__ == '__main__':
    tree = Tree(r'pse/terngrad.pse.txt')
    for token in tree.tokens:
        logging.debug(token['str'])
    tree.block_code()
