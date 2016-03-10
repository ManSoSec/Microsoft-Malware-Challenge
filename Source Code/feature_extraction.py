
import os, gzip, re
import entropy
import strings
from collections import defaultdict
from handle_io import io
from settings import *


def byte_1gram(byte_code):
    OneByte = [0]*16**2
    for row in byte_code:
        codes = row[:-2].split()[1:]
        # Convert code to 1byte
        OneByteCode = []
        for i in codes:
            if i != '??':
                OneByteCode += [int(i,16)]

        # Calculate the frequency of 1byte
        for i in OneByteCode:
                    OneByte[i] += 1
    return OneByte
#onegram_numba = autojit(byte_1gram, nopython=True)

def byte_2gram(byte_code):
    twoByte = [0]*16**4
    for row in byte_code:
        codes = row[:-2].split()[1:]
        codes_2g = codes[:-1]
        for i in range(len(codes_2g)):
            codes_2g[i]+= codes[i+1]

        twoByteCode = []
        for i in codes_2g:
            if '??' not in i:
                twoByteCode += [int(i,16)]

        for i in twoByteCode:
            twoByte[i] += 1
    return twoByte


def byte_meta_data(file_path, file):

    #filesize
    meta_data = []
    statinfo = os.stat(file_path)
    fileSize = statinfo.st_size
    meta_data.append(fileSize)

    #StartAddress
    firstLine = file.readline().split()
    offset = firstLine[0]
    dec = int(offset, 16)
    meta_data.append(dec)

    return meta_data


def byte_entropy(file_name):
    ents = entropy.get_feats([entropy.get_entropy_features(file_name)])
    #ent = entropy.get_entropy_features(file_name)
    #print len(ents[0])
    return ents[0]


def byte_string_lengths(file_name):
    strs_len = strings.extract_length([strings.get_strings(file_name)])
    #print strs_len
    return strs_len[0].tolist()
    #pass


import mahotas
import mahotas.features

def byte_image1(byte_code):
    img_feat = []
    img = byte_make_image(byte_code)
    #img = mahotas.imread(img.im)
    features = mahotas.features.haralick(img)
    for i in range(len(features)):
        for j in range(len(features[0])):
            img_feat.append(features[i][j])
    return img_feat

from mahotas.features.lbp import lbp

def byte_image2(byte_code):
    img = byte_make_image(byte_code)
    spoints = lbp(img,10,10,ignore_zeros=False)
    return spoints.tolist()


from math import log
import numpy as np

def byte_make_image(byte_code):
    img_array=[]
    for row in byte_code:
        xx=row.split()
        if len(xx)!=17:
            continue
        img_array.append([int(i,16) if i!='??' else 0 for i in xx[1:] ])
    img_array = np.array(img_array)
    if img_array.shape[1]!=16:
        assert(False)
    b=int((img_array.shape[0]*16)**(0.5))
    b=2**(int(log(b)/log(2))+1)
    a=int(img_array.shape[0]*16/b)
    img_array=img_array[:a*b/16,:]
    img_array=np.reshape(img_array,(a,b))
    #img_array = np.uint8(img_array)
    #im = Image.fromarray(img_array)
    return img_array


###################################################


def asm_meta_data(file_path, asm_code):

    #filesize
    meta_data = []
    statinfo = os.stat(file_path)
    fileSize = statinfo.st_size
    meta_data.append(fileSize)

    #StartAddress
    loc = 0
    for row in asm_code:
        loc += 1
    meta_data.append(loc)

    return meta_data


def asm_symbols(asm_code):
    symbols = [0]*7
    for row in asm_code:
        if '*' in row:
            symbols[0] += 1
        if '-' in row:
            symbols[1] += 1
        if '+' in row:
            symbols[2] += 1
        if '[' in row:
            symbols[3] += 1
        if ']' in row:
            symbols[4] += 1
        if '@' in row:
            symbols[5] += 1
        if '?' in row:
            symbols[6] += 1

    return symbols


def asm_registers(asm_code):
    registers = ['edx','esi','es','fs','ds','ss','gs','cs','ah','al',
                 'ax','bh','bl','bx','ch','cl','cx','dh','dl','dx',
                 'eax','ebp','ebx','ecx','edi','esp']
    registers_values = [0]*len(registers)
    for row in asm_code:
        parts = row.replace(',',' ').replace('+',' ').replace('*',' ').replace('[',' ').replace(']',' ') \
                    .replace('-',' ').split()
        for register in registers:
            registers_values[registers.index(register)] += parts.count(register)
    return registers_values


def asm_opcodes(asm_code):
    opcodes = ['add','al','bt','call','cdq','cld','cli','cmc','cmp','const','cwd','daa','db'
                ,'dd','dec','dw','endp','ends','faddp','fchs','fdiv','fdivp','fdivr','fild'
                ,'fistp','fld','fstcw','fstcwimul','fstp','fword','fxch','imul','in','inc'
                ,'ins','int','jb','je','jg','jge','jl','jmp','jnb','jno','jnz','jo','jz'
                ,'lea','loope','mov','movzx','mul','near','neg','not','or','out','outs'
                ,'pop','popf','proc','push','pushf','rcl','rcr','rdtsc','rep','ret','retn'
                ,'rol','ror','sal','sar','sbb','scas','setb','setle','setnle','setnz'
                ,'setz','shl','shld','shr','sidt','stc','std','sti','stos','sub','test'
                ,'wait','xchg','xor']
    opcodes_values = [0]*len(opcodes)
    for row in asm_code:
        #flag = False
        #for opcode in opcodes:
        #    if opcode in row:
        #        flag = True
        #        break
        #if flag == False:
        #    continue
        #if '"' in row:
        #    crow = row[1:-1].replace("\"","")
        #else:
        #    crow = row

        #parts = re.split(" |\r|,|/|\'|:|;|\+|\*|[|]|_|\-|\(|\)|\[|\]|\.|\\\\",row)
        # It is faster than the above regex
        #parts = row.replace("\r",' ').replace(',',' ').replace('/',' ').replace("'",' ').replace(':',' ') \
        #            .replace(';',' ').replace('+',' ').replace('*',' ').replace('[',' ').replace(']',' ') \
        #            .replace('_',' ').replace('-',' ').replace('(',' ').replace(')',' ').replace('.',' ') \
        #            .replace('\\',' ').split()
        parts = row.split()

        for opcode in opcodes:
            if opcode in parts:
                opcodes_values[opcodes.index(opcode)] += 1 #parts.count(opcode)
                break
    return opcodes_values


def asm_APIs(asm_code, apis):
    apis_values = [0]*len(apis)
    for row in asm_code:
        #parts = row.replace(',',' ').replace('+',' ').replace('*',' ').replace('[',' ').replace(']',' ') \
        #            .replace('-',' ').split()
        for i in range(len(apis)):
            if apis[i] in row:
                apis_values[i] += 1 #parts.count(opcode)
                break
    return apis_values


def asm_misc(asm_code):

    keywords = ['Virtual','Offset','loc','Import','Imports','var','Forwarder','UINT','LONG','BOOL','WORD','BYTES','large','short','dd','db','dw','XREF','ptr','DATA','FUNCTION','extrn','byte','word','dword','char','DWORD','stdcall','arg','locret','asc','align','WinMain','unk','cookie','off','nullsub','DllEntryPoint','System32','dll','CHUNK','BASS','HMENU','DLL','LPWSTR','void','HRESULT','HDC','LRESULT','HANDLE','HWND','LPSTR','int','HLOCAL','FARPROC','ATOM','HMODULE','WPARAM','HGLOBAL','entry','rva','COLLAPSED','config','exe','Software','CurrentVersion','__imp_','INT_PTR','UINT_PTR','---Seperator','PCCTL_CONTEXT','__IMPORT_','INTERNET_STATUS_CALLBACK','.rdata:','.data:','.text:','case','installdir','market','microsoft','policies','proc','scrollwindow','search','trap','visualc','___security_cookie','assume','callvirtualalloc','exportedentry','hardware','hkey_current_user','hkey_local_machine','sp-analysisfailed','unableto']

    keywords_values = [0]*len(keywords)
    for row in asm_code:
        #parts = row.replace(',',' ').replace('+',' ').replace('*',' ').replace('[',' ').replace(']',' ') \
        #            .replace('-',' ').split()
        for i in range(len(keywords)):
            if keywords[i] in row:
                keywords_values[i] += 1 #parts.count(opcode)
                break
    return keywords_values


def asm_sections(asm_code):
    section_names = []
    for row in asm_code:
        section_name = [row[0:np.core.defchararray.index(row, ':')]]
        if section_name != 'HEADER':
            section_names += section_name

    known_sections = ['.text', '.data', '.bss', '.rdata', '.edata', '.idata', '.rsrc', '.tls', '.reloc']
    sections_values = [0]*24
    unknown_sections = []
    unknown_lines = 0
    number_of_sections = len(section_names)

    for section in section_names:

        if section in known_sections:
            section_index = known_sections.index(section)
            sections_values[section_index] += 1
        else:
            unknown_sections.append(section)
            unknown_lines += 1

    uni_section_names_len = len(np.unique(section_names))
    uni_unknown_section_names_len = len(np.unique(unknown_sections))
    uni_known_section_names_len = 0
    for i in range(0,8):
        if sections_values[i] != 0:
            uni_known_section_names_len += 1

    sections_values[9] = uni_section_names_len
    sections_values[10] = uni_unknown_section_names_len
    sections_values[11] = unknown_lines

    for i in range(0,8):
        sections_values[i + 12] = float(sections_values[i])/ number_of_sections

    sections_values[21] = float(uni_known_section_names_len) / uni_section_names_len
    sections_values[22] = float(uni_unknown_section_names_len) / uni_section_names_len
    sections_values[23] = float(unknown_lines) / number_of_sections

    return sections_values, section_names


def asm_data_define(asm_code):
    #data_define_inst = ['dd', 'db', 'dw']
    #data_dict = {}
    # data_dict = defaultdict(list)
    # for row in asm_code:
    #     parts = row.replace(',', ' ').split()
    #     section_name = parts[0].split(':')
    #     data_define = ''
    #     data_define_v = 0
    #     data_define_i = 0
    #     found = 0
    #     db_0 = False
    #     for inst in data_define_inst:
    #         for i in range(len(parts)):
    #             if inst == parts[i]:
    #                 data_define_i = i + 1
    #                 data_define = inst
    #                 found = 1
    #                 break
    #         if found == 1:
    #             break
    #     data_define_v = len(parts) - data_define_i
    #
    #     if data_define == 'db':
    #         if parts[data_define_i] == '0':
    #             db_0 = True
    #
    #     if data_define == 'dd':
    #
    #         #if section_name in data_dict.keys():
    #         data_dict[section_name].append((data_define,data_define_v))
    #         #else:
    #         #    data_dict[section_name] = [data_define+','+str(data_define_v)]
    #
    #     print row
    #     print parts
    #     print data_define_v, data_define
    #Allcounter  = 0
    dbCounter = 0
    ddCounter = 0
    dwCounter = 0
    dcCounter = 0
    db0Counter = 0
    dbN0Counter = 0

    all = 0
    text = 0
    rdata = 0
    data = 0
    dd_text = 0
    db_text = 0
    dd_rdata = 0
    db3_rdata = 0
    db3_data = 0
    db3_all = 0
    dd4 = 0
    dd5 = 0
    dd6 = 0

    all = 0

    text = 0
    rdata = 0
    data = 0
    idata = 0
    # NotdataNottext
    NdNt = 0

    db3_idata = 0
    db3_text = 0
    db3_rsrc = 0
    db3_NdNt = 0
    db3_all = 0
    db3_zero = 0

    dd_text = 0
    db_text = 0
    dd_rdata = 0
    db3_data = 0
    db3_all = 0
    dd4_NdNt = 0
    dd5_NdNt = 0
    dd6_NdNt = 0

    for row in asm_code:
        RowItems = row.split()
        Section = row.split(':')[0]
        RowComma = row.split(',')

        all += 1
        dbCounter += RowItems.count('db')
        ddCounter += RowItems.count('dd')
        dwCounter += RowItems.count('dw')
        if len(RowItems)>3:
            if RowItems[1]=='00' and RowItems[2]=='db':
                db0Counter += 1

        if Section=='.text':
            text +=1
            dd_text += RowItems.count('dd')
            db_text += RowItems.count('db')
        elif Section=='.rdata':
            rdata +=1
            dd_rdata += RowItems.count('dd')
            if len(RowItems) == 4 or len(RowItems) == 6:
                if RowItems[2] == 'db':
                    db3_rdata += RowItems.count('db')
        elif Section=='.data':
            data +=1
            if len(RowItems) == 4 or len(RowItems) == 6:
                if RowItems[2] == 'db':
                    db3_data += RowItems.count('db')

        if len(RowItems) == 4 or len(RowItems) == 6:
                if RowItems[2] == 'db':
                    db3_all += RowItems.count('db')
        elif Section=='.idata':
            idata +=1
            if len(RowItems) == 4 or len(RowItems) == 6:
                if RowItems[2] == 'db':
                    db3_idata += 1
        else:
            NdNt += 1
            if len(RowItems) == 4 or len(RowItems) == 6:
                if RowItems[2] == 'db':
                    db3_NdNt += 1

            if len(RowComma)==4:
                dd4_NdNt += RowItems.count('dd')

            if len(RowComma)==5:
                dd5_NdNt += RowItems.count('dd')

            if len(RowComma)==6:
                dd6_NdNt += RowItems.count('dd')

        if len(RowComma)==4:
            dd4 += RowItems.count('dd')

        if len(RowComma)==5:
            dd5 += RowItems.count('dd')

        if len(RowComma)==6:
            dd6 += RowItems.count('dd')

        if len(RowItems) == 4 or len(RowItems) == 6:
            if RowItems[2] == 'db':
                db3_all += 1
                if RowItems[1] == '00':
                    db3_zero += 1


    dcCounter = dbCounter + ddCounter + dwCounter
    db_por = float(dbCounter)/all
    dd_por = float(ddCounter)/all
    dw_por = float(dwCounter)/all
    dc_por = float(dcCounter)/all
    db0_por = dbN0_por = 0
    if dbCounter!=0:
        db0_por = float(db0Counter)/dbCounter
        dbN0_por = float(dbCounter - db0Counter)/dbCounter

    ############################

    Res_dd_text = 0
    Res_db_text = 0
    Res_dd_rdata = 0
    Res_db3_rdata = 0
    Res_db3_data = 0

    if text!=0:
        Res_dd_text = float(dd_text)/text
        Res_db_text = float(db_text)/text

    if rdata!=0:
        Res_dd_rdata = float(dd_rdata)/rdata
        Res_db3_rdata = float(db3_rdata)/rdata

    if data!=0:
        Res_db3_data = float(db3_data)/data

    Res_db3_all = float(db3_all)/all

    Res_dd4_all = float(dd4)/all
    Res_dd5_all = float(dd5)/all
    Res_dd6_all = float(dd6)/all

    Output = [Res_dd_text,Res_db_text,Res_dd_rdata,Res_db3_rdata \
              , Res_db3_data, Res_db3_all, dd4, dd5, dd6 \
              , Res_dd4_all, Res_dd5_all, Res_dd6_all]


    Res_db3_idata = 0
    Res_db3_NdNt = 0
    Res_dd4_NdNt = 0
    Res_dd5_NdNt = 0
    Res_dd6_NdNt = 0
    Res_db3_all_zero = 0

    if idata!=0:
        Res_db3_idata = float(db3_idata)/idata

    if NdNt!=0:
        Res_db3_NdNt = float(db3_NdNt)/NdNt
        Res_dd4_NdNt = float(dd4_NdNt)/NdNt
        Res_dd5_NdNt = float(dd5_NdNt)/NdNt
        Res_dd6_NdNt = float(dd6_NdNt)/NdNt

    if db3_all!=0:
        Res_db3_all_zero = float(db3_zero)/db3_all

    Output2 = [Res_db3_idata, Res_db3_NdNt, Res_dd4_NdNt \
              , Res_dd5_NdNt, Res_dd6_NdNt, Res_db3_all_zero ]


    #print [db_por, dd_por, dw_por, dc_por, db0_por, dbN0_por] + Output + Output2
    return [db_por, dd_por, dw_por, dc_por, db0_por, dbN0_por] + Output + Output2

