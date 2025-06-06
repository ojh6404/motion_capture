from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union
from pyvirtualdisplay import Display
import os
import pyrender
import trimesh
import numpy as np
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams

from motion_capture.utils import HAND_COLORS


def convert_opencv_to_pytorch3d(
    verts: Union[np.ndarray, torch.Tensor],
    R: Optional[Union[np.ndarray, torch.Tensor]] = None,
    T: Optional[Union[np.ndarray, torch.Tensor]] = None,
):
    """
    Convert OpenCV coordinates to PyTorch3D coordinates.

    Args:
        verts: (N, 3) vertices in OpenCV system
        R: (3, 3) rotation matrix in OpenCV coordinates
        T: (3,) translation vector in OpenCV coordinates

    Returns:
        verts_pytorch3d: (N, 3) PyTorch3D coordinates
        R_pytorch3d: (3, 3) rotation matrix in PyTorch3D coordinates
        T_pytorch3d: (3,) translation vector in PyTorch3D coordinates
    """
    # Transformation matrix to convert OpenCV coordinates to PyTorch3D coordinates
    transform = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    # Vertices
    if isinstance(verts, torch.Tensor):
        transform = torch.from_numpy(transform).float().to(verts.device)
        verts_pytorch3d = verts @ transform.T
    else:
        verts_pytorch3d = verts @ transform.T

    results = [verts_pytorch3d]

    # Rotation matrix
    if R is not None:
        if isinstance(R, torch.Tensor):
            transform_torch = torch.from_numpy(transform).float().to(R.device)
            R_pytorch3d = transform_torch @ R @ transform_torch.T
        else:
            R_pytorch3d = transform @ R @ transform.T
        results.append(R_pytorch3d)

    # Translation vector
    if T is not None:
        if isinstance(T, torch.Tensor):
            transform_torch = torch.from_numpy(transform).float().to(T.device)
            T_pytorch3d = transform_torch @ T
        else:
            T_pytorch3d = transform @ T
        results.append(T_pytorch3d)

    return results if len(results) > 1 else results[0]


class Renderer(ABC):
    def __init__(self, K: np.ndarray, width: int, height: int):
        super(Renderer, self).__init__()
        assert K.shape == (3, 3), "K must be of shape (3, 3) of ndarray or torch.Tensor"
        self.width = width
        self.height = height
        self.K = K

        # Initialize camera and renderer
        self.init()

    @abstractmethod
    def render(
        self,
        verts: List[np.ndarray],
        faces: List[np.ndarray],
        colors: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the given vertices and faces.

        Args:
            verts (List[np.ndarray]): List of vertices to render.
            faces (List[np.ndarray]): List of faces for the meshes.
            colors (Optional[List[Tuple[float, float, float]]]): List of RGB colors for each mesh. If None, default color is used.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Rendered RGBA image, mask, and depth map.
        """
        pass

    @abstractmethod
    def init(self):
        """
        Initialize the renderer, camera, and scene.
        This method should be called in the constructor.
        """
        pass

    def __del__(self):
        """
        Clean up resources when the renderer is deleted.
        """
        if hasattr(self, "display"):
            self.display.stop()
            del self.display
        if hasattr(self, "renderer"):
            del self.renderer
        if hasattr(self, "camera"):
            del self.camera
        if hasattr(self, "lights"):
            del self.lights


class PyrenderRenderer(Renderer):
    def __init__(self, K: np.ndarray, width: int = 640, height: int = 480):
        if "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL for OpenGL context creation
        """
        Initialize the PyrenderRenderer.

        Args:
            K (np.ndarray): Camera intrinsic matrix of shape (3, 3).
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
        """

        # Initialize display for headless rendering
        self.display = Display(visible=0, size=(width, height))
        self.display.start()
        super(PyrenderRenderer, self).__init__(K, width, height)

    def init(self):
        # Initialize camera and renderer
        self.camera_pose = np.eye(4)
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2]
        )
        self.lights = self.create_lights()
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)

    def verts_to_mesh(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        colors: Tuple[float, float, float] = HAND_COLORS[0],
    ) -> trimesh.Trimesh:
        colors = np.array([(*colors, 1.0)] * verts.shape[0])
        mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render(
        self,
        verts: List[np.ndarray],
        faces: List[np.ndarray],
        colors: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the given vertices and faces using pyrender.

        Args:
            verts (np.ndarray): List of vertices to render. Shape: (V, 3) where V is the number of vertices.
            faces (np.ndarray): List of faces for the meshes. Shape: (F, 3)
            colors (Optional[List[Tuple[float, float, float]]]): List of RGB colors for each mesh. If None, default color is used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rendered RGBA image and depth map.
        """
        assert len(verts) == len(faces), "Number of vertices and faces must match"
        if colors is None:
            colors = HAND_COLORS[: len(verts)]

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Add meshes to the scene
        for i, (vert, face, color) in enumerate(zip(verts, faces, colors)):
            mesh = pyrender.Mesh.from_trimesh(self.verts_to_mesh(verts=vert, faces=face, colors=color))
            scene.add(mesh, f"mesh_{i}")

        # Create camera node and add it to pyRender scene
        scene.add(self.camera, pose=self.camera_pose)

        # Add lights to the scene
        for node in self.lights:
            scene.add_node(node)

        # Render the scene
        rgba, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        mask = rgba[..., 3] > 0
        return rgba, mask, depth

    def create_lights(self) -> List[pyrender.Node]:
        """
        Return raymond light nodes for the scene.

        Returns:
            List[pyrender.Node]: List of pyrender nodes with directional lights.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix)
            )

        return nodes


class Pytorch3DRenderer(Renderer):
    """
    Renderer using PyTorch3D for rendering meshes.

    Args:
        K (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
    """

    def __init__(
        self, K: np.ndarray, width: int, height: int, device: str = "cuda", from_opencv: bool = True
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.from_opencv = from_opencv
        super(Pytorch3DRenderer, self).__init__(K, width, height)

    def init(self):
        self.K = torch.from_numpy(self.K).float().to(self.device)
        self.camera_pose = torch.eye(4, device=self.device)

        self.camera = PerspectiveCameras(
            R=self.camera_pose[:3, :3].unsqueeze(0),  # (1, 3, 3)
            T=self.camera_pose[:3, 3].unsqueeze(0),  # (1, 3)
            focal_length=((self.K[0, 0], self.K[1, 1]),),
            principal_point=((self.K[0, 2], self.K[1, 2]),),
            image_size=((self.height, self.width),),
            device=self.device,
            in_ndc=False,
        )

        # TODO : Add lights, currently using default point lights
        self.lights = PointLights(
            device=self.device,
            location=[[2.0, 2.0, -2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.5, 0.5, 0.5]],
            specular_color=[[0.3, 0.3, 0.3]],
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=True,
        )

        # Blend parameters for soft shading
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # Create renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=self.lights,
                materials=Materials(device=self.device),
                blend_params=blend_params,
            ),
        )

    def render(
        self,
        verts: List[Union[np.ndarray, torch.Tensor]],
        faces: List[Union[np.ndarray, torch.Tensor]],
        colors: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the given vertices and faces using PyTorch3D.

        Args:
            verts (List[Union[np.ndarray, torch.Tensor]]): List of vertices to render. Each vertex array should be of shape (V, 3) where V is the number of vertices.
            faces (List[Union[np.ndarray, torch.Tensor]]): List of faces for the meshes. Each face array should be of shape (F, 3).
            colors (Optional[List[Tuple[float, float, float]]]): List of RGB colors for each mesh. If None, default color is used.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Rendered RGBA image, mask, and depth map.
        """
        # Mesh
        if colors is None:
            colors = HAND_COLORS[: len(verts)]
        assert len(verts) == len(faces) == len(colors), "verts, faces, and colors must have the same length"

        meshes = []
        for vert, face, color in zip(verts, faces, colors):
            assert vert.shape[1] == 3, "verts must be of shape (V, 3)"
            mesh = self.verts_to_mesh(verts=vert, faces=face, colors=color)
            meshes.append(mesh)
        meshes = join_meshes_as_scene(meshes)  # Combine all meshes into a single scene

        # Rendering
        images = self.renderer(meshes)

        # Rasterization
        fragments = self.renderer.rasterizer(meshes)

        # Extract RGBA image
        rgba = (images[0].cpu().numpy() * 255).astype(np.uint8)  # (H, W, 4)

        # Extract mask
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)
        mask = (pix_to_face[0, ..., 0] >= 0).cpu().numpy().astype(np.float32)

        # Extract depth information
        zbuf = fragments.zbuf[0, ..., 0]  # (H, W)
        depth = zbuf.cpu().numpy()
        valid_depth = depth > 0  # valid depth
        depth[~valid_depth] = np.inf  # Set invalid depth to infinity

        return rgba, mask, depth

    def verts_to_mesh(
        self,
        verts: Union[np.ndarray, torch.Tensor],
        faces: Union[np.ndarray, torch.Tensor],
        colors: Tuple[float, float, float] = HAND_COLORS[0],
    ) -> Meshes:
        """
        Convert vertices and faces to a PyTorch3D mesh.

        Args:
            verts (Union[np.ndarray, torch.Tensor]): Vertices of the mesh.
            faces (Union[np.ndarray, torch.Tensor]): Faces of the mesh.
            colors (Tuple[float, float, float]): RGB color for the mesh.

        Returns:
            Meshes: PyTorch3D mesh object.
        """
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts).float().to(self.device)
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).long().to(self.device)

        if self.from_opencv:
            verts = convert_opencv_to_pytorch3d(verts)

        colors = (
            torch.tensor(colors, dtype=torch.float32, device=self.device).unsqueeze(0).expand(len(verts), -1)
        )
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))  # (1, V, 3)
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        return mesh
