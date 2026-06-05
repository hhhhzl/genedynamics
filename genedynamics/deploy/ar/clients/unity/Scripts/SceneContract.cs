using System;

namespace CorridorTwin
{
    // Data model mirroring the scene_server JSON contract. Plain [Serializable]
    // classes with public fields so Unity's JsonUtility can parse them.
    // (plan_xy_scene is float[][], which JsonUtility can't handle — parse it
    //  separately if you need the ghost path; it's optional for rendering.)

    [Serializable]
    public class Se2 // T_world_scene: scene -> world (planar)
    {
        public float x;
        public float y;
        public float yaw;
    }

    [Serializable]
    public class Corridor
    {
        public float corridor_width;
        public float corridor_length;
        public float[] start_pos;   // [x, y] scene frame
        public float[] goal_pos;    // [x, y] scene frame
        public float wall_y_min;
        public float wall_y_max;
    }

    [Serializable]
    public class Obstacle
    {
        public string name;
        public string shape;        // "box" | "sphere" | "qc"
        // box bounds (scene frame, meters)
        public float x_min, x_max, y_min, y_max, z_min, z_max;
        // sphere / quarter-circle
        public float cx, cy, radius;
        public float qc_clip_sign;  // +1 entry / -1 exit (qc only)
    }

    [Serializable]
    public class SceneContractMsg
    {
        public int schema_version;
        public string frame;        // "vicon_world"
        public string units;        // "m_rad"
        public int revision;
        public string scene_preset;
        public Se2 T_world_scene;
        public Corridor corridor;
        public Obstacle[] obstacles;
        public int seq;
        public double server_time;

        public static SceneContractMsg Parse(string json)
        {
            return UnityEngine.JsonUtility.FromJson<SceneContractMsg>(json);
        }
    }
}
