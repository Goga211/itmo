transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/fec_saylinx_board_top.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/morr_l2.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/root3.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/mult.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/seg_display_ctrl.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/hex_to_7seg.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/lfsr8.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/crc8.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/debounce.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/bist_ctrl.v}
vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/bist_wrapper.v}

vlog -vlog01compat -work work +incdir+C:/Schema/lab3 {C:/Schema/lab3/bist_demo_tb.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cycloneive_ver -L rtl_work -L work -voptargs="+acc"  bist_demo_tb

do C:/Schema/lab3/bist_demo.do
