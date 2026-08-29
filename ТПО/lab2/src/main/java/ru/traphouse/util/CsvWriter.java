package ru.traphouse.util;

import ru.traphouse.system.FunctionSystem;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.function.DoubleUnaryOperator;

public class CsvWriter {

    /**
     * Записывает значения функции в CSV-файл.
     *
     * @param path     путь к файлу
     * @param from     начало диапазона X
     * @param to       конец диапазона X
     * @param step     шаг X
     * @param function функция для вычисления
     * @param header   заголовок второй колонки
     */
    public static void write(String path, double from, double to, double step,
                              DoubleUnaryOperator function, String header) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("X;" + header);
            for (double x = from; x <= to + 1e-12; x += step) {
                String valStr;
                try {
                    valStr = String.valueOf(function.applyAsDouble(x));
                } catch (ArithmeticException e) {
                    valStr = "undefined(" + e.getMessage() + ")";
                }
                pw.printf("%.6f;%s%n", x, valStr);
            }
        }
    }

    public static void write(String path, double from, double to, double step,
                              FunctionSystem system) throws IOException {
        write(path, from, to, step, system::calculate, "F(x)");
    }
}
