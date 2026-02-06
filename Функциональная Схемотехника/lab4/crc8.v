`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Модуль CRC8 (Cyclic Redundancy Check - 8 bit)
// 
// Полином: x^8 + x^7 + x^3 + x + 1 (эквивалентно 1 + x + x^3 + x^7 + x^8)
// В hex: 0x8B (без старшего бита x^8)
//
// Алгоритм: побитовая обработка входных данных
// На каждом такте обрабатывается один бит входных данных
//-----------------------------------------------------------------------------
module crc8 #(
    parameter [7:0] POLY = 8'b10001011,  // x^7 + x^3 + x + 1 (без x^8)
    parameter [7:0] INIT = 8'h00         // Начальное значение CRC
)(
    input  wire       clk_i,
    input  wire       rst_i,
    input  wire       clear_i,    // Сброс CRC в начальное значение
    input  wire       en_i,       // Разрешение обработки данных
    input  wire [7:0] data_i,     // Входные данные (обрабатываются за 8 тактов)
    input  wire       byte_en_i,  // Обработать целый байт за один такт
    output wire [7:0] crc_o       // Текущее значение CRC
);

    reg [7:0] crc_reg;
    
    // Функция для вычисления CRC за один такт (обработка целого байта)
    function [7:0] crc8_byte;
        input [7:0] crc_in;
        input [7:0] data_in;
        reg [7:0] crc_tmp;
        integer i;
        begin
            crc_tmp = crc_in ^ data_in;
            for (i = 0; i < 8; i = i + 1) begin
                if (crc_tmp[7])
                    crc_tmp = {crc_tmp[6:0], 1'b0} ^ POLY;
                else
                    crc_tmp = {crc_tmp[6:0], 1'b0};
            end
            crc8_byte = crc_tmp;
        end
    endfunction

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            crc_reg <= INIT;
        end else if (clear_i) begin
            crc_reg <= INIT;
        end else if (en_i && byte_en_i) begin
            // Обработка целого байта за один такт
            crc_reg <= crc8_byte(crc_reg, data_i);
        end
    end

    assign crc_o = crc_reg;

endmodule



