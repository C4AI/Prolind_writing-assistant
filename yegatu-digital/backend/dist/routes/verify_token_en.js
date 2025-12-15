import jwt from "jsonwebtoken";
import { data } from "../shared/data.js";
export const tokenMiddlewareEn = (req, res, next) => {
    const token = req.headers["authorization"];
    if (!token) {
        res.status(401);
        res.statusMessage = "Invalid Token";
        res.send();
    }
    else {
        try {
            const decoded = jwt.verify(token, data.tokenKey);
            req.body.username = decoded.username;
            req.token = token;
        }
        catch (error) {
            res.status(401);
            res.statusMessage = "Invalid Token";
            res.send();
            return;
        }
        next();
    }
};
export const verifyTokenEn = (req, res) => {
    const token = req.headers["authorization"];
    if (!token) {
        res.status(401);
        res.statusMessage = "Invalid Token";
        res.send();
        return;
    }
    else {
        jwt.verify(token, data.tokenKey, (error) => {
            if (error) {
                res.status(401);
                res.statusMessage = "Invalid Token";
                res.send();
                return;
            }
            else {
                res.status(200);
                res.send();
            }
        });
    }
};
//# sourceMappingURL=verify_token_en.js.map