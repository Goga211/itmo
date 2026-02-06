`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Модуль антидребезга кнопки с синхронизацией
// 
// Включает:
// 1. Двухтриггерную синхронизацию для устранения метастабильности
// 2. Счётчик-фильтр для устранения дребезга
// 3. Детектор фронта для формирования импульса нажатия
//-----------------------------------------------------------------------------
module debounce #(
    parameter DEBOUNCE_CYCLES = 1_000_000  // Количество тактов для фильтрации (~20мс при 50МГц)
)(
    input  wire clk_i,
    input  wire rst_i,
    input  wire btn_i,      // Вход кнопки (может быть асинхронным)
    output wire btn_o,      // Отфильтрованное состояние кнопки
    output wire btn_posedge_o,  // Импульс на нажатие (передний фронт)
    output wire btn_negedge_o   // Импульс на отпускание (задний фронт)
);

    // Ширина счётчика
    localparam CNT_WIDTH = $clog2(DEBOUNCE_CYCLES + 1);

    //-------------------------------------------------------------------------
    // Двухтриггерная синхронизация (устранение метастабильности)
    //-------------------------------------------------------------------------
    reg sync_ff1, sync_ff2;
    
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            sync_ff1 <= 1'b0;
            sync_ff2 <= 1'b0;
        end else begin
            sync_ff1 <= btn_i;
            sync_ff2 <= sync_ff1;
        end
    end
    
    wire btn_sync = sync_ff2;

    //-------------------------------------------------------------------------
    // Счётчик-фильтр дребезга
    //-------------------------------------------------------------------------
    reg [CNT_WIDTH-1:0] counter;
    reg btn_stable;
    
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            counter <= 0;
            btn_stable <= 1'b0;
        end else begin
            if (btn_sync != btn_stable) begin
                // Входной сигнал отличается от стабильного - считаем
                if (counter >= DEBOUNCE_CYCLES - 1) begin
                    btn_stable <= btn_sync;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end else begin
                // Сигнал совпадает - сброс счётчика
                counter <= 0;
            end
        end
    end

    //-------------------------------------------------------------------------
    // Детектор фронтов
    //-------------------------------------------------------------------------
    reg btn_prev;
    
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i)
            btn_prev <= 1'b0;
        else
            btn_prev <= btn_stable;
    end

    assign btn_o = btn_stable;
    assign btn_posedge_o = btn_stable & ~btn_prev;   // Передний фронт (нажатие)
    assign btn_negedge_o = ~btn_stable & btn_prev;   // Задний фронт (отпускание)

endmodule



