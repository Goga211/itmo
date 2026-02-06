package ru.goga.lab2;

public class Parameters {
    final private float x;
    final private int y;
    final private int r;


    public Parameters(float x, int y, int r) {
        this.x = x;
        this.y = y;
        this.r = r;
    }

    public float getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getR() {
        return r;
    }


    @Override
    public String toString() {
        return "Parameters{" +
                "x=" + x +
                ", y=" + y +
                ", r=" + r +
                '}';
    }
}


