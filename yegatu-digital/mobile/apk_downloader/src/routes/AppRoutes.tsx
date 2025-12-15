import {
  createBrowserRouter,
  Navigate,
  Outlet,
  RouterProvider,
} from "react-router-dom";
import LoginPage from "../components/pages/LoginPage/LoginPage";
import Spinner from "../components/common/Spinner/Spinner";
import DownloadPage from "../components/pages/DownloadPage/DownloadPage";
import { useAuth } from "../hooks/use_auth";
import { useVerifyToken } from "../hooks/use_verify_token";
import { useLoading } from "../hooks/use_loading";

// function that protects the routes that require authentication and redirects to the login page if the token does not exist
const ProtectedRoute = () => {
  const { token } = useAuth();
  useVerifyToken();

  if (!token) {
    return <Navigate to="/" />;
  }

  return <Outlet />;
};

// function used to create routes of the application.
const AppRoutes = () => {
  const { isLoading } = useLoading();

  // routes that does not require authentication
  const publicRoutes = [
    {
      path: "/",
      element: <LoginPage />,
    },
  ];

  // routes that require authentication (using the ProtectedRoute function)
  const authenticatedRoutes = [
    {
      path: "/",
      element: <ProtectedRoute />,
      children: [
        {
          path: "/download",
          element: (
            <>
              {isLoading ? (
                <main className="flex flex-col items-center h-full gap-2 bg-gray-500">
                  <Spinner />
                </main>
              ) : (
                <DownloadPage />
              )}
            </>
          ),
        },
      ],
    },
  ];

  const router = createBrowserRouter([...publicRoutes, ...authenticatedRoutes]);

  return <RouterProvider router={router} />;
};

export default AppRoutes;
