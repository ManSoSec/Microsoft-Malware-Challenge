
from settings import *

def header_byte_1gram():
    colnames = []
    for i in range(16**2):
        temp = hex(i)[2:]
        if len(str(temp))==1:
            temp = "0"+str(temp)
        colnames += ['byte_1G_'+ temp]
    return colnames


def header_byte_2grams():
    colnames = []
    colnames += ['byte_2G_'+hex(i)[2:] for i in range(16**4)]
    return colnames


def header_byte_meta_data():
    colnames = ['byte_filesize','byte_offset']
    return colnames


def header_byte_img1():
    colnames = []
    for i in range(52):
        colnames += ['byte_Img1_'+str(i)]
    return colnames


def header_byte_img2():
    colnames = []
    for i in range(108):
        colnames += ['byte_Img2_'+str(i)]
    return colnames


def header_byte_str_len():
    colnames = ['string_len_counts_' + str(x) for x in range(1,100)] + [
        'string_len_counts_0_10',
        'string_len_counts_10_30',
        'string_len_counts_30_60',
        'string_len_counts_60_90',
        'string_len_counts_0_100',
        'string_len_counts_100_150',
        'string_len_counts_150_250',
        'string_len_counts_250_400',
        'string_len_counts_400_600',
        'string_len_counts_600_900',
        'string_len_counts_900_1300',
        'string_len_counts_1300_2000',
        'string_len_counts_2000_3000',
        'string_len_counts_3000_6000',
        'string_len_counts_6000_15000',
        'string_total_len',
        'string_ratio'
    ]
    return colnames


def header_byte_entropy():
    colnames = []
    st = ['mean','var','median','max','min','max-min']

    colnames.extend( ['ent_q_diffs_' + str(x) for x in range(21) ])
    colnames.extend( ['ent_q_diffs_' + x for x in st])

    colnames.extend( ['ent_q_diff_diffs_' + str(x) for x in range(21) ])
    colnames.extend( ['ent_q_diff_diffs_' + x for x in st])

    for i in range(4):
        colnames.extend( ['ent_q_diff_block_' + str(i) + '_' + str(x) for x in range(21) ])
        colnames.extend( ['ent_q_diff_diffs_'+ str(i) + '_' + x for x in st])

    colnames.extend( ['ent_p_' + str(x) for x in range(20) ])
    colnames.extend( ['ent_p_diffs_' + str(x) for x in range(20) ])

    return colnames

######################################################################


def header_asm_meta_data():
    colnames = ['asm_md_filesize','asm_md_loc']
    return colnames


def header_asm_sym():
    symbols = ['Star','Dash','Plus','Bracket_Open','Bracket_Close','AtSign','Question']
    colnames = ['asm_symb_'+reg for reg in symbols]
    return colnames


def header_asm_registers():
    registers = ['edx','esi','es','fs','ds','ss','gs','cs','ah','al',
                 'ax','bh','bl','bx','ch','cl','cx','dh','dl','dx',
                 'eax','ebp','ebx','ecx','edi','esp']

    colnames = ['asm_regs_'+reg for reg in registers]
    return colnames


def header_asm_opcodes():
    opcodes = ['add','al','bt','call','cdq','cld','cli','cmc','cmp','const','cwd','daa','db'
                ,'dd','dec','dw','endp','ends','faddp','fchs','fdiv','fdivp','fdivr','fild'
                ,'fistp','fld','fstcw','fstcwimul','fstp','fword','fxch','imul','in','inc'
                ,'ins','int','jb','je','jg','jge','jl','jmp','jnb','jno','jnz','jo','jz'
                ,'lea','loope','mov','movzx','mul','near','neg','not','or','out','outs'
                ,'pop','popf','proc','push','pushf','rcl','rcr','rdtsc','rep','ret','retn'
                ,'rol','ror','sal','sar','sbb','scas','setb','setle','setnle','setnz'
                ,'setz','shl','shld','shr','sidt','stc','std','sti','stos','sub','test'
                ,'wait','xchg','xor']

    colnames = ['asm_opcodes_'+op for op in opcodes]
    return colnames

def header_asm_sections():
    kown_sections = ['.text','.data','.bss', '.rdata','.edata','.idata', '.rsrc','.tls','.reloc']
    colnames = kown_sections + ['Num_Sections', 'Unknown_Sections', 'Unknown_Sections_lines']
    colnames += ['.text_por','.data_por','.bss_por', '.rdata_por','.edata_por',
                 '.idata_por', '.rsrc_por','.tls_por','.reloc_por']
    colnames += ['known_Sections_por', 'Unknown_Sections_por', 'Unknown_Sections_lines_por']
    return colnames


def header_asm_data_define():
    colnames = ['db_por','dd_por','dw_por','dc_por','db0_por','dbN0_por','dd_text',
                'db_text','dd_rdata','db3_rdata','db3_data','db3_all','dd4','dd5',
                'dd6','dd4_all','dd5_all','dd6_all']
    return colnames


def header_asm_apis():
    defined_apis = io.read_all_lines(APIS_PATH)
    colnames = defined_apis[0].split(',')
    return colnames
