FROM node:20-alpine AS build-backend

WORKDIR /app/backend

COPY package.json ./
RUN npm install


COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app/backend

COPY --from=build-backend /app/backend/dist /app/backend/dist
COPY --from=build-backend /app/backend/node_modules /app/backend/node_modules
COPY --from=build-backend /app/backend/package.json /app/backend/package.json

ARG COUCH_DB_URL
ENV COUCH_DB_URL=${COUCH_DB_URL}

EXPOSE 3000

CMD ["node", "/app/backend/dist/index.js"]