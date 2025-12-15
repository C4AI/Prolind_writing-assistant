FROM node:20-alpine AS build-frontend

WORKDIR /app/frontend

COPY package.json ./
RUN npm install

ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

COPY . .
RUN npm run build

FROM nginx:alpine

WORKDIR /app

COPY --from=build-frontend /app/frontend/dist /usr/share/nginx/html

COPY default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
