import { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { service } from "../../shared/couchdb_service.js";

export const deleteFeedback = async (req: Request, res: Response) => {
  let feedback_id;
  if (!req.body.feedback_id) {
    res.status(400);
    res.send();
    return;
  }
  feedback_id = req.body.feedback_id;

  try {
    try {
      let rev = "";
      let docId = "";
      console.log(rev);
      console.log(feedback_id);
      service
        .getDocument({
          db: "assistente-nheengatu",
          docId: "feedback:" + feedback_id,
        })
        .then((response) => {
          console.log("passou");
          if (!response.result._rev || !response.result._id) {
            res.status(400);
            res.send();
            return;
          }
          rev = response.result._rev!;
          docId = response.result._id!;
        })
        .then(() => {
          service
            .deleteDocument({
              db: "assistente-nheengatu",
              docId: docId,
              rev: rev,
            })
            .then((response) => {
              console.log(
                "Cloudant response:" + JSON.stringify(response.result)
              );
              res.json({ feedback_id: feedback_id });
            });
        });
    } catch (err) {
      console.log("Error:", err);
    }
  } catch (err) {
    console.error("Error:", err);
  }
};
