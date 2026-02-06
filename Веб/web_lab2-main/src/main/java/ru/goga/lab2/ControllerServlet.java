package ru.goga.lab2;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.util.logging.Logger;

@WebServlet(name = "ControllerServlet", value = "/controller-servlet")
public class ControllerServlet extends HttpServlet {

    private static final Logger logger = Logger.getLogger(ControllerServlet.class.getName());
    private Validator validator;


    @Override
    public void init() {
        this.validator = new Validator();
        logger.info("Создан Validator");
    }

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            Parameters parameters = parseAndSetParameters(request);
            logger.info("Парсинг параметров прошёл успешно:\n" + parameters);
            validator.validate(parameters.getX(), parameters.getY(), parameters.getR());
            logger.info("Валидация прошла успешно");
            request.setAttribute("parameters", parameters);

            logger.info("Переадресация на area-check-servlet");
            getServletContext().getRequestDispatcher("/area-check-servlet").forward(request, response);

        } catch (ValidateException e) {
            logger.warning(e.getMessage());
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
        } catch (IllegalArgumentException e) {
            logger.warning("Неверный формат одной или нескольких переменных, либо отсутствуют значения");
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
        } catch (Exception e) {
            logger.severe("Неожиданная ошибка: " + e.getMessage());
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Внутренняя ошибка сервера");
        }
    }

    private Parameters parseAndSetParameters(HttpServletRequest request) {
        String x = request.getParameter("x");
        String y = request.getParameter("y");
        String r = request.getParameter("r");

        if (x == null || y == null || r == null) {
            throw new IllegalArgumentException("Один или несколько параметров отсутствуют");
        }

        try {
            float xValue = Float.parseFloat(x);
            int yValue = Integer.parseInt(y);
            int rValue = Integer.parseInt(r);

            return new Parameters(xValue, yValue, rValue);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Неверный формат одного или нескольких параметров: x = " + x + ", y = " + y + ", r = " + r);
        }
    }


}


