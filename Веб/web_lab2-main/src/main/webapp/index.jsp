<%@ page import="ru.goga.lab2.ResultList" %>
<%@ page import="java.util.List" %>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>ЛР №2</title>
    <link rel="stylesheet" href="css/site.css" type="text/css">
    <script src="js/index.js"></script>
</head>
<body>

<header>Антипин Григорий Викторович, P3332, 562374</header>

<table>
    <tr>
        <!-- Графика (SVG) в первой ячейке -->
        <td id="coordinate-plate">
            <svg width="500" height="500" xmlns="http://www.w3.org/2000/svg" id="plate">
                <!-- Ось X -->
                <line id="axis-x" x1="50" y1="250" x2="450" y2="250" stroke="silver" stroke-width="2"></line>
                <!-- Ось Y -->
                <line id="axis-y" x1="250" y1="50" x2="250" y2="450" stroke="silver" stroke-width="2"></line>

                <!-- Стрелки -->
                <polygon id="arrow-x" points="450,245 450,255 460,250" fill="silver"></polygon>
                <polygon id="arrow-y" points="245,50 255,50 250,40" fill="silver"></polygon>

                <%-- 1 четверть: четверть круга радиусом R/2 --%>
                <path id="arc" d="M250,250 L300,250 A50,50 0 0,0 250,200 Z" fill="white"></path>

                <%-- 2 четверть: треугольник --%>
                <polygon id="triangle" points="250,250 200,250 250,200" fill="white"></polygon>

                <%-- 3 четверть: квадрат --%>
                <rect id="rect" x="150" y="250" width="100" height="100" fill="white"></rect>

                <!-- Подписи осей -->
                <text x="260" y="50" fill="white">Y</text>
                <text x="450" y="240" fill="white">X</text>

                <g id="ticks" stroke="silver" stroke-width="2" fill="white" font-size="12">
                    <line x1="150" y1="247" x2="150" y2="253"></line>
                    <text x="150" y="268" text-anchor="middle">-5</text>

                    <line x1="170" y1="247" x2="170" y2="253"></line><text x="170" y="268" text-anchor="middle">-4</text>
                    <line x1="190" y1="247" x2="190" y2="253"></line><text x="190" y="268" text-anchor="middle">-3</text>
                    <line x1="210" y1="247" x2="210" y2="253"></line><text x="210" y="268" text-anchor="middle">-2</text>
                    <line x1="230" y1="247" x2="230" y2="253"></line><text x="230" y="268" text-anchor="middle">-1</text>

                    <line x1="250" y1="245" x2="250" y2="255"></line>

                    <line x1="270" y1="247" x2="270" y2="253"></line><text x="270" y="268" text-anchor="middle">1</text>
                    <line x1="290" y1="247" x2="290" y2="253"></line><text x="290" y="268" text-anchor="middle">2</text>
                    <line x1="310" y1="247" x2="310" y2="253"></line><text x="310" y="268" text-anchor="middle">3</text>
                    <line x1="330" y1="247" x2="330" y2="253"></line><text x="330" y="268" text-anchor="middle">4</text>

                    <line x1="350" y1="247" x2="350" y2="253"></line>
                    <text x="350" y="268" text-anchor="middle">5</text>

                    <line x1="247" y1="350" x2="253" y2="350"></line><text x="238" y="354" text-anchor="end">-5</text>
                    <line x1="247" y1="330" x2="253" y2="330"></line><text x="238" y="334" text-anchor="end">-4</text>
                    <line x1="247" y1="310" x2="253" y2="310"></line><text x="238" y="314" text-anchor="end">-3</text>
                    <line x1="247" y1="290" x2="253" y2="290"></line><text x="238" y="294" text-anchor="end">-2</text>
                    <line x1="247" y1="270" x2="253" y2="270"></line><text x="238" y="274" text-anchor="end">-1</text>

                    <line x1="245" y1="250" x2="255" y2="250"></line><text x="238" y="254" text-anchor="end">0</text>

                    <line x1="247" y1="230" x2="253" y2="230"></line><text x="238" y="234" text-anchor="end">1</text>
                    <line x1="247" y1="210" x2="253" y2="210"></line><text x="238" y="214" text-anchor="end">2</text>
                    <line x1="247" y1="190" x2="253" y2="190"></line><text x="238" y="194" text-anchor="end">3</text>
                    <line x1="247" y1="170" x2="253" y2="170"></line><text x="238" y="174" text-anchor="end">4</text>
                    <line x1="247" y1="150" x2="253" y2="150"></line><text x="238" y="154" text-anchor="end">5</text>
                </g>

        <%
    List<ResultList> resultList = (List<ResultList>) application.getAttribute("resultList");
    if (resultList != null) {
        for (ResultList result1 : resultList) {
            if (result1.getResult()){%>
                <circle cx=<%=250+ 20 *result1.getX()%> cy=<%=250 - 20*result1.getY()%> r="2" fill="green" visibility="visible"></circle>
                <%
                        }
            else { %>
                <circle cx=<%=250+ 20 *result1.getX()%> cy=<%=250 - 20*result1.getY()%> r="2" fill="red" visibility="visible"></circle>
        <%}
                        }
                    }%>
            </svg>

        </td>

        <!-- Форма в отдельной строке, в одной ячейке -->
        <td id="input">
            <div id="error" hidden></div>
            <form action="/" method="GET" id="data-form">

                <!-- Ввод X -->
                <!-- Выбор радиуса X -->
                <fieldset id="legend-x">
                    <legend> Выберите радиус X:</legend>
                    <label><input type="radio" name="x" value="-2"> -2</label>
                    <label><input type="radio" name="x" value="-1.5"> -1.5</label>
                    <label><input type="radio" name="x" value="-1"> -1</label>
                    <label><input type="radio" name="x" value="-0.5"> -0.5</label>
                    <label><input type="radio" name="x" value="0"> 0</label>
                    <label><input type="radio" name="x" value="0.5"> 0.5</label>
                    <label><input type="radio" name="x" value="1"> 1</label>
                    <label><input type="radio" name="x" value="1.5"> 1.5</label>
                    <label><input type="radio" name="x" value="2"> 2</label>

                </fieldset>

                <!-- Ввод Y -->
                <fieldset id="legend-y">
                    <label for="y">Введите Y (от -5 до 3):</label>
                    <input type="number" id="y" name="y" min="-5" max="3" step="0.1" required>
                </fieldset>

                <!-- Выбор радиуса R -->
                <fieldset id="legend-r">
                    <legend> Выберите радиус R:</legend>
                    <label><input type="radio" name="r" value="1"> 1</label>
                    <label><input type="radio" name="r" value="2"> 2</label>
                    <label><input type="radio" name="r" value="3"> 3</label>
                    <label><input type="radio" name="r" value="4"> 4</label>
                    <label><input type="radio" name="r" value="5" checked> 5</label>
                </fieldset>

                <!-- Кнопка отправки -->
                <button id="submit" type="submit">Проверить</button>
            </form>
        </td>

        <td id="result">
            <table id="result-table">
                <tr>
                    <th id="result-x">X</th>
                    <th id="result-y">Y</th>
                    <th id="result-r">R</th>
                    <th id="result-time">Time</th>
                    <th id="result-now">Now</th>
                    <th id="result-answer">Result</th>
                </tr>
                <%

                    if (resultList != null) {
                        for (ResultList result : resultList) {
                %>
                <tr>
                    <td><%=result.getX()%>
                    </td>
                    <td><%=result.getY()%>
                    </td>
                    <td><%=result.getR()%>
                    </td>
                    <td><%=result.getCompleteTime()%> нс
                    </td>
                    <td><%=result.getTime()%>
                    </td>

                    <td><%=result.getResult()%>
                    </td>
                        <%
                            }
                        }%>

            </table>
        </td>
    </tr>
</table>

</body>
</html>
