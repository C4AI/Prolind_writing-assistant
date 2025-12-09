import jwt, { JwtPayload } from "jsonwebtoken";
import { data } from "../shared/data.js";
import { NextFunction, Request, Response } from "express";

export const tokenMiddleware = (
  req: Request & { username?: string; token?: string },
  res: Response,
  next: NextFunction
) => {
  const token: string = req.headers["authorization"] as string;
  if (!token) {
    res.status(401);
    res.statusMessage = "Token inválido";
    res.send();
    return;
  } else {
    try {
      const decoded = jwt.verify(token, data.tokenKey) as JwtPayload;

      req.body.username = decoded.username;
      req.token = token;
    } catch (error) {
      res.status(401);
      res.statusMessage = "Token inválido";
      res.send();
      return;
    }

    next();
  }
};

export const verifyToken = (req: Request, res: Response) => {
  const token: string = req.headers["authorization"] as string;

  if (!token) {
    res.status(401);
    res.statusMessage = "Token inválido";
    res.send();
    return;
  } else {
    jwt.verify(token, data.tokenKey, (error) => {
      if (error) {
        res.status(401);
        res.statusMessage = "Token inválido";
        res.send();
        return;
      } else {
        res.status(200);
        res.send();
      }
    });
  }
};
