#include <igl/directed_edge_orientations.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/PI.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/readTGF.h>
#include <igl/boundary_conditions.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/jet.h>
#include <igl/normalize_row_sums.h>
#include <igl/viewer/Viewer.h>
#include <igl/bbw.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

/*

 Get weights from tutorial 403

* tutorial 403 operates on a mesh file.
Most libigl headers are tailored to operate on a generic triangle mesh 
stored in:
	* n-by-3 matrix of vertex positions V 
	* m-by-3 matrix of triangle indices F.

Pre-computation of weights, then do LBS or dQS like this file is doing.

Decrease number of triangles in mesh. [done: use eight-bar.obj]

Bar with triangles on the surface
With tetgen, the surfaces will be filled with objects on the inside 
Deformations to the bar should also apply to the tetrahedrals inside.


Exporting tetgen files into dmat -> can read into MATLAB to draw.

There is a chance that the Bounded biharmonic weight calculation already
does this -- tetrahedralises the mesh and computes for all points, then only
returns the surface points -- check if true.

Tetrahedral "volume" is required for physics simulation later.

*/



typedef 
  std::vector<Eigen::Quaterniond,Eigen::aligned_allocator<Eigen::Quaterniond> >
  RotationList;

const Eigen::RowVector3d sea_green(70./255.,252./255.,167./255.);
Eigen::MatrixXd V,W,C,U,M;
Eigen::MatrixXi F,BE;
Eigen::VectorXi P;
std::vector<RotationList > poses;
double anim_t = 0.0;
double anim_t_dir = 0.015;
bool use_dqs = false;
bool recompute = true;

bool pre_draw(igl::viewer::Viewer & viewer)
{
  using namespace Eigen;
  using namespace std;
  if(recompute)
  {
    // Find pose interval
    const int begin = (int)floor(anim_t)%poses.size();
    const int end = (int)(floor(anim_t)+1)%poses.size();
    const double t = anim_t - floor(anim_t);

    // Interpolate pose and identity
    RotationList anim_pose(poses[begin].size());
    for(int e = 0;e<poses[begin].size();e++)
    {
      anim_pose[e] = poses[begin][e].slerp(t,poses[end][e]);
    }
    // Propogate relative rotations via FK to retrieve absolute transformations
    RotationList vQ;
    vector<Vector3d> vT;
    igl::forward_kinematics(C,BE,P,anim_pose,vQ,vT);
    const int dim = C.cols();
    MatrixXd T(BE.rows()*(dim+1),dim);
    for(int e = 0;e<BE.rows();e++)
    {
      Affine3d a = Affine3d::Identity();
      a.translate(vT[e]);
      a.rotate(vQ[e]);
      T.block(e*(dim+1),0,dim+1,dim) =
        a.matrix().transpose().block(0,0,dim+1,dim);
    }
    // Compute deformation via LBS as matrix multiplication
    if(use_dqs)
    {
      igl::dqs(V,W,vQ,vT,U);
    }else
    {
      U = M*T;
    }

    // Also deform skeleton edges
    MatrixXd CT;
    MatrixXi BET;
    igl::deform_skeleton(C,BE,T,CT,BET);
    
    viewer.data.set_vertices(U);
    viewer.data.set_edges(CT,BET,sea_green);
    viewer.data.compute_normals();
    if(viewer.core.is_animating)
    {
      anim_t += anim_t_dir;
    }
    else
    {
      recompute=false;
    }
  }
  return false;
}

bool key_down(igl::viewer::Viewer &viewer, unsigned char key, int mods)
{
  recompute = true;
  switch(key)
  {
    case 'D':
    case 'd':
      use_dqs = !use_dqs;
      return true;
    case ' ':
      viewer.core.is_animating = !viewer.core.is_animating;
      return true;
  }
  return false;
}

int main(int argc, char *argv[])
{

/*
  using namespace Eigen;
  using namespace std;
  igl::readOBJ("/home/xiuyan/Desktop/eight-bar.obj",V,F);
  U=V;
  igl::readTGF("/home/xiuyan/Desktop/middle.tgf",C,BE);

  // retrieve parents for forward kinematics
  igl::directed_edge_parents(BE,P);
  RotationList rest_pose;
  igl::directed_edge_orientations(C,BE,rest_pose);
  poses.resize(4,RotationList(4,Quaterniond::Identity()));

  // poses[1] // twist
  const Quaterniond twist(AngleAxisd(igl::PI,Vector3d(1,0,0)));
  poses[1][2] = rest_pose[2]*twist*rest_pose[2].conjugate();
  const Quaterniond bend(AngleAxisd(-igl::PI*0.7,Vector3d(0,0,1)));
  poses[3][2] = rest_pose[2]*bend*rest_pose[2].conjugate();

  igl::readDMAT("/home/xiuyan/Desktop/middle-weights.dmat",W);
  igl::lbs_matrix(V,W,M);
*/
// instead of these weights (above), use BBW

  using namespace Eigen;
  using namespace std;
  igl::readOBJ("/home/xiuyan/Desktop/eight-bar.obj",V,F);
  U=V;
  igl::readTGF("/home/xiuyan/Desktop/middle.tgf",C,BE);

  // retrieve parents for forward kinematics
  igl::directed_edge_parents(BE,P);
  RotationList rest_pose;
  igl::directed_edge_orientations(C,BE,rest_pose);
  poses.resize(4,RotationList(4,Quaterniond::Identity()));

  // poses[1] // twist
  const Quaterniond twist(AngleAxisd(igl::PI,Vector3d(1,0,0)));
  poses[1][2] = rest_pose[2]*twist*rest_pose[2].conjugate();
  const Quaterniond bend(AngleAxisd(-igl::PI*0.7,Vector3d(0,0,1)));
  poses[3][2] = rest_pose[2]*bend*rest_pose[2].conjugate();


  // List of boundary indices (aka fixed value indices into VV)
  VectorXi b;
  // List of boundary conditions of each weight function
  MatrixXd bc;
  igl::boundary_conditions(V,F,C,VectorXi(),BE,MatrixXi(),b,bc);

  // compute BBW weights matrix
  igl::BBWData bbw_data;
  // only a few iterations for sake of demo
  bbw_data.active_set_params.max_iter = 100;
  bbw_data.verbosity = 2;
  if(!igl::bbw(V,F,b,bc,bbw_data,W))
  {
    return false;
  }

  //MatrixXd Vsurf = V.topLeftCorner(F.maxCoeff()+1,V.cols());
  //MatrixXd Wsurf;
  //if(!igl::bone_heat(Vsurf,F,C,VectorXi(),BE,MatrixXi(),Wsurf))
  //{
  //  return false;
  //}
  //W.setConstant(V.rows(),Wsurf.cols(),1);
  //W.topLeftCorner(Wsurf.rows(),Wsurf.cols()) = Wsurf = Wsurf = Wsurf = Wsurf;

  // Normalize weights to sum to one
  igl::normalize_row_sums(W,W);
  // precompute linear blend skinning matrix
  igl::lbs_matrix(V,W,M);


///////


  // Plot the mesh with pseudocolors
  igl::viewer::Viewer viewer;
  viewer.data.set_mesh(U, F);
  viewer.data.set_edges(C,BE,sea_green);
  viewer.core.show_lines = false;
  viewer.core.show_overlay_depth = false;
  viewer.core.line_width = 1;
  viewer.core.trackball_angle.normalize();
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core.is_animating = false;
  viewer.core.camera_zoom = 2.5;
  viewer.core.animation_max_fps = 30.;
  cout<<"Press [d] to toggle between LBS and DQS"<<endl<<
    "Press [space] to toggle animation"<<endl;
  viewer.launch();
}
