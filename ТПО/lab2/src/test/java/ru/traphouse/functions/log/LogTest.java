package ru.traphouse.functions.log;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LogTest {
    private static final double EPS = 1e-6;
    private Log2 log2;
    private Log3 log3;
    private Log5 log5;
    private Log10 log10;

    @BeforeEach
    void setUp() {
        Ln ln = new Ln(1e-10);
        log2  = new Log2(ln);
        log3  = new Log3(ln);
        log5  = new Log5(ln);
        log10 = new Log10(ln);
    }

    @Test
    void testLog2AtTwo() {
        assertEquals(1.0, log2.calculate(2.0), EPS);
    }

    @Test
    void testLog2AtFour() {
        assertEquals(2.0, log2.calculate(4.0), EPS);
    }

    @Test
    void testLog3AtThree() {
        assertEquals(1.0, log3.calculate(3.0), EPS);
    }

    @Test
    void testLog3AtNine() {
        assertEquals(2.0, log3.calculate(9.0), EPS);
    }

    @Test
    void testLog5AtFive() {
        assertEquals(1.0, log5.calculate(5.0), EPS);
    }

    @Test
    void testLog10AtTen() {
        assertEquals(1.0, log10.calculate(10.0), EPS);
    }

    @Test
    void testLog10AtHundred() {
        assertEquals(2.0, log10.calculate(100.0), EPS);
    }

    @Test
    void testLogAtOne() {
        assertEquals(0.0, log2.calculate(1.0), EPS);
        assertEquals(0.0, log3.calculate(1.0), EPS);
        assertEquals(0.0, log10.calculate(1.0), EPS);
    }

    @Test
    void testLog10VsJava() {
        assertEquals(Math.log10(7.0), log10.calculate(7.0), EPS);
    }
}
