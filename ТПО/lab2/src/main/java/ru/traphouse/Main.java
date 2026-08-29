package ru.traphouse;

import ru.traphouse.functions.log.*;
import ru.traphouse.functions.trig.*;
import ru.traphouse.system.FunctionSystem;
import ru.traphouse.system.LogSystem;
import ru.traphouse.system.TrigSystem;
import ru.traphouse.util.CsvWriter;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        double epsilon = 1e-10;

        // --- Тригонометрические модули ---
        Sin sin = new Sin(epsilon);
        Cos cos = new Cos(sin);
        Cot cot = new Cot(sin, cos);
        Csc csc = new Csc(sin);

        // --- Логарифмические модули ---
        Ln    ln    = new Ln(epsilon);
        Log2  log2  = new Log2(ln);
        Log3  log3  = new Log3(ln);
        Log5  log5  = new Log5(ln);
        Log10 log10 = new Log10(ln);

        // --- Системы ---
        TrigSystem     trigSystem = new TrigSystem(sin, cot, csc);
        LogSystem      logSystem  = new LogSystem(ln, log2, log3, log5, log10);
        FunctionSystem system     = new FunctionSystem(trigSystem, logSystem);

        // --- CSV-вывод ---
        CsvWriter.write("trig_output.csv",   -3 * Math.PI, -0.1, 0.1,
                x -> trigSystem.calculate(x), "TrigSystem(x)");

        CsvWriter.write("log_output.csv",    0.1, 10.0, 0.1,
                x -> logSystem.calculate(x),  "LogSystem(x)");

        CsvWriter.write("system_output.csv", -3 * Math.PI, 10.0, 0.2,
                system::calculate,            "F(x)");

        System.out.println("CSV files written: trig_output.csv, log_output.csv, system_output.csv");
    }
}
