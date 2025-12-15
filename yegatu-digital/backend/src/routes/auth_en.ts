import { Request, Response } from "express";
import { data } from "../shared/data.js";
import jwt from "jsonwebtoken";
import { service } from "../shared/couchdb_service.js";

export const authEn = (req: Request, res: Response) => {
  let username = req.body.username;
  const password = req.body.password;
  let appId = req.body.app_id;

  console.log("just to test the workflow");

  // **REMOVE
  if (!appId) {
    appId = 1;
  }

  const users = data.users;

  username = username.toLowerCase().trim();
  let response;
  if (username in users) {
    if (users[username].password == password.trim()) {
      if (!users[username].app_id.includes(appId)) {
        res.status(401);
        res.send();
        return;
      }
      const token = jwt.sign({ username: username }, data.tokenKey, {
        expiresIn: data.userAuthTimeout, // represented in seconds
      });
      users[username].token = token;

      response = {
        success: true,
        token: token,
        username: username,
      };

      // register the user login in database
      const docId = Date.now();
      const dateTime = new Date().toLocaleString("pt-BR", {
        timeZone: "Brazil/East",
      });
      const doc = {
        _id: "login:" + username + "_" + docId,
        created: dateTime,
        username: username,
        token: token,
      };

      try {
        service
          .postDocument({
            db: "assistente-nheengatu",
            document: doc,
          })
          .then((response) => {
            console.log("Cloudant response:" + JSON.stringify(response.result));
          });
      } catch (err) {
        console.error("Error:", err);
      }
    } else {
      res.status(401);
      res.statusMessage = "Incorrect Password";
      res.send();
    }
  } else {
    res.status(401);
    res.statusMessage = "Invalid User";
    res.send();
  }
  res.json(response!);
};
