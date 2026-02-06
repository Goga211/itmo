`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// 8-битный LFSR (Linear Feedback Shift Register)
// Генератор псевдослучайных чисел
// 
// Полиномы задаются параметром POLY:
//   LFSR1: x^8 + x^6 + x^4 + x^3 + 1 -> POLY = 8'b01011000 (биты 6,4,3)
//   LFSR2: x^8 + x^7 + x^6 + x^3 + 1 -> POLY = 8'b11000100 (биты 7,6,3)
//
// Реализация: Fibonacci (внешний) LFSR
// Новый бит = XOR старшего бита с битами, указанными в POLY
//-----------------------------------------------------------------------------
module lfsr8 #(
    parameter [7:0] POLY = 8'b01011000,  // Полином (без x^8 и x^0)
    parameter [7:0] SEED = 8'h01         // Начальное значение (не должно быть 0)
)(
    input  wire       clk_i,
    input  wire       rst_i,
    input  wire       en_i,      // Разрешение сдвига
    input  wire       load_i,    // Загрузка начального значения
    output wire [7:0] data_o     // Текущее значение регистра
);

    reg [7:0] lfsr_reg;
    wire feedback;

    // Обратная связь: XOR старшего бита с битами из полинома
    // Для полинома x^8 + x^6 + x^4 + x^3 + 1:
    //   feedback = reg[7] ^ reg[5] ^ reg[3] ^ reg[2]
    // Общая формула: feedback = reg[7] ^ (reg & POLY)
    assign feedback = lfsr_reg[7] ^ 
                      (POLY[6] & lfsr_reg[5]) ^
                      (POLY[5] & lfsr_reg[4]) ^
                      (POLY[4] & lfsr_reg[3]) ^
                      (POLY[3] & lfsr_reg[2]) ^
                      (POLY[2] & lfsr_reg[1]) ^
                      (POLY[1] & lfsr_reg[0]) ^
                      (POLY[7] & lfsr_reg[6]);

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            lfsr_reg <= SEED;
        end else if (load_i) begin
            lfsr_reg <= SEED;
        end else if (en_i) begin
            // Сдвиг влево, новый бит вставляется в младший разряд
            lfsr_reg <= {lfsr_reg[6:0], feedback};
        end
    end

    assign data_o = lfsr_reg;

endmodule



