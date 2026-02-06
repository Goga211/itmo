package ru.goga.lab2;

public class ResultList {
    private final float x;
    private final int y;
    private final int r;
    private final long completeTime;
    private final String time;
    private final boolean result;

    public ResultList(float x, int y, int r, long completeTime, String time, boolean result) {
        this.x = x;
        this.y = y;
        this.r = r;
        this.completeTime = completeTime;
        this.time = time;

        this.result = result;

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
    public long getCompleteTime() {
        return completeTime;
    }

    public String getTime() {
        return time;
    }


    public boolean getResult() {
        return result;
    }

    @Override
    public String toString() {
        return "ResultList{" +
                "x=" + x +
                ", y=" + y +
                ", r=" + r +
                ", completeTime=" + completeTime +
                ", time='" + time + '\'' +
                ", result=" + result +
                '}';
    }
}


