package ru.goga.lab2;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import static ru.goga.lab2.DotChecker.checkDot;
import static ru.goga.lab2.UrlConstants.RESULT_PAGE_URL;

@WebServlet(name = "AreaCheckServlet", value = "/area-check-servlet")
public class AreaCheckServlet extends HttpServlet {

    private static final long MIN_EXECUTION_TIME_NS = 1;
    private static final Logger logger = Logger.getLogger(AreaCheckServlet.class.getName());

    @Override
    public void init() {
        if (getServletContext().getAttribute("resultList") == null) {
            getServletContext().setAttribute("resultList", new ArrayList<ResultList>());
        }
    }

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            Parameters parameters = (Parameters) request.getAttribute("parameters");
            if (parameters == null) {
                throw new IllegalArgumentException("Отсутствует Parameters аргумент");
            }

            logger.info("Получен Parameters\n" + parameters);

            long startTime = System.nanoTime();
            boolean result = checkDot(parameters.getX(), parameters.getY(), parameters.getR());
            long endTime = System.nanoTime();

            long executionTime = Math.max(endTime - startTime, MIN_EXECUTION_TIME_NS);
            String formattedNow = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

            List<ResultList> resultList = getResultListFromContext();
            ResultList resultDto = new ResultList(parameters.getX(), parameters.getY(), parameters.getR(), executionTime, formattedNow, result);
            resultList.add(resultDto);

            logger.info("Добавлен результат\n" + resultDto);
            logger.info("Переадресация на страницу результата");

            response.sendRedirect(request.getContextPath() + RESULT_PAGE_URL);



        } catch (IllegalArgumentException e) {
            logger.warning(e.getMessage());
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
        }
    }



    @SuppressWarnings("unchecked")
    private List<ResultList> getResultListFromContext() {
        List<ResultList> resultList = (List<ResultList>) getServletContext().getAttribute("resultList");
        if (resultList == null) {
            resultList = new ArrayList<>();
            getServletContext().setAttribute("resultList", resultList);
        }
        return resultList;
    }
}


