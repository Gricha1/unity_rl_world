using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class AgentToTarget : Agent {
    [SerializeField] private Transform goal;
    [SerializeField] private Transform wall;
    [SerializeField] private GameObject checkpoint_obj;
    [SerializeField] private Transform snowman;
    [SerializeField] private Transform agent_1;

    // Dataset
    [SerializeField] private GameObject dynamicObst;

    [SerializeField] private float movementSpeed = 80f;
    [SerializeField] private float rotateSpeed = 120f;
    [SerializeField] private float dynamicMovementSpeed = 40f;
    [SerializeField] private int env_type = 2;
    // 1 - snowman, 2 - slide, 3 - maze
    [SerializeField] private int maxSteps = 3000;
    [SerializeField] private bool eval_env = false;
    [SerializeField] private bool dynamic_obst_eval_env = false;
    
    // agent_type = 1 - agent
    // agent_type = 2 - snowman 
    [SerializeField] private int agent_type = 1;
    [SerializeField] private bool long_horzion_tasks = false;
    [SerializeField] private bool continue_train_task = false;
    [SerializeField] private bool descrete_actions = false;
    [SerializeField] private bool add_slide = false;
    private int stepCount;
    private Rigidbody rb;
    private BoxCollider checkpoint_box_col;
    private Transform checkpoint;
    private int checkPointMeetCounts = 0;
    private bool eval_env_while_train;
    private bool reverse_eval_direction = false;
    private bool isTriggeredCheckPoint = false;
    private bool isTriggeredOtherAgent = false;
    private Vector3 left_up_agent;
    private Vector3 right_down_agent;
    private Vector3 SampleRandomVector(Vector3 start, Vector3 end) {
        float start_x = start.x;
        float start_y = start.y;
        float start_z = start.z;
        float goal_x = end.x;
        float goal_y = end.y;
        float goal_z = end.z;
        
        float x_sampled = Random.Range(Mathf.Min(start_x, goal_x), Mathf.Max(start_x, goal_x));
        float y_sampled = Random.Range(Mathf.Min(start_y, goal_y), Mathf.Max(start_y, goal_y));
        float z_sampled = Random.Range(Mathf.Min(start_z, goal_z), Mathf.Max(start_z, goal_z));
        return new Vector3(x_sampled, y_sampled, z_sampled); // Сэмплируем вектор
    } 
    private Vector3 AddPertrubXZ(Vector3 current) {
        return new Vector3(current.x + Random.Range(-5f, 5f), current.y, current.z + Random.Range(-5f, 5f)); // Сэмплируем вектор
    } 
    private List<Vector3> starts = new List<Vector3>();
    private List<Vector3> goals = new List<Vector3>();
    private List<Vector3> checkpoints = new List<Vector3>();
    private List<Vector3> dynamic_obsts_start_poses = new List<Vector3>();
    private List<Vector2> dynamic_obsts_movings = new List<Vector2>();
    private List<GameObject> episodeDynamicObstacles = new List<GameObject>();
    private Vector3 start_eval_pose;
    private Vector3 goal_eval_pose;
    private int task_id;


    public override void Initialize() {
        rb = GetComponent<Rigidbody>();
        checkpoint_box_col = checkpoint_obj.GetComponent<BoxCollider>();
        checkpoint = checkpoint_obj.transform;
        if (env_type == 3) {
            // get train dataset
            float x1 = -133.2604f;
            float x2 = -36.9f;
            float x3 = 127.1f;
            float y = 28.89999f;
            float z1 = 128.174f;
            float z2 = 53.17395f;
            float z3 = -20.82605f;
            float z4 = -95.82605f;
            float z4_last = -133.7342f;

            List<float> xs = new List<float>{x1, x2, x3};
            List<float> zs = new List<float>{z1, z2, z3, z4};
            // short horizon tasks
            for (int i_x=0; i_x<xs.Count; i_x++) {
                for (int i_z=0; i_z<zs.Count; i_z++) {
                    starts.Add(new Vector3(xs[i_x], y, zs[i_z]));
                    if (i_x == 0) {
                        goals.Add(new Vector3(xs[i_x+1], y, zs[i_z]));
                    }
                    else if (i_x == 1) {
                        goals.Add(new Vector3(xs[i_x+1], y, zs[i_z]));
                        starts.Add(new Vector3(xs[i_x], y, zs[i_z]));
                        goals.Add(new Vector3(xs[i_x-1], y, zs[i_z]));
                    }
                    else if (i_x == 2) {
                        goals.Add(new Vector3(xs[i_x-1], y, zs[i_z]));
                        if (i_z + 1 < zs.Count) {
                            starts.Add(new Vector3(xs[i_x], y, zs[i_z]));
                            goals.Add(new Vector3(xs[i_x], y, zs[i_z+1]));
                        }
                    }
                }
            }
            // long horizon tasks
            if (long_horzion_tasks) {
                starts.Add(new Vector3(xs[0], y, zs[0]));
                goals.Add(new Vector3(xs[2], y, zs[0]));
                starts.Add(new Vector3(xs[0], y, zs[0]));
                goals.Add(new Vector3(xs[2], y, zs[1]));

                starts.Add(new Vector3(xs[2], y, zs[0]));
                goals.Add(new Vector3(xs[1], y, zs[1]));
                starts.Add(new Vector3(xs[2], y, zs[0]));
                goals.Add(new Vector3(xs[0], y, zs[1]));

                starts.Add(new Vector3(xs[2], y, zs[1]));
                goals.Add(new Vector3(xs[0], y, zs[1]));
                starts.Add(new Vector3(xs[2], y, zs[1]));
                goals.Add(new Vector3(xs[0], y, zs[2]));

                starts.Add(new Vector3(xs[0], y, zs[1]));
                goals.Add(new Vector3(xs[1], y, zs[2]));
                starts.Add(new Vector3(xs[0], y, zs[1]));
                goals.Add(new Vector3(xs[2], y, zs[2]));

                starts.Add(new Vector3(xs[2], y, zs[2]));
                goals.Add(new Vector3(xs[1], y, zs[3]));
                starts.Add(new Vector3(xs[2], y, zs[2]));
                goals.Add(new Vector3(xs[0], y, zs[3]));
            }            
            

            // reversed tasks
            List<Vector3> current_starts = new List<Vector3>(starts);
            List<Vector3> current_goals = new List<Vector3>(goals);
            starts.AddRange(current_goals);
            goals.AddRange(current_starts);


            // get checkpoints for validation
            start_eval_pose = new Vector3(x1, y, z1);
            goal_eval_pose = new Vector3(x1, y, z4_last);
            checkpoints.Add(new Vector3(x2, y, z1));
            checkpoints.Add(new Vector3(x3, y, z1));
            checkpoints.Add(new Vector3(x3, y, z2));
            checkpoints.Add(new Vector3(x2, y, z2));
            checkpoints.Add(new Vector3(x1, y, z2));
            checkpoints.Add(new Vector3(x1, y, z3));
            checkpoints.Add(new Vector3(x2, y, z3));
            checkpoints.Add(new Vector3(x3, y, z3));
            checkpoints.Add(new Vector3(x3, y, z4));
            checkpoints.Add(new Vector3(x2, y, z4));
            checkpoints.Add(new Vector3(x1, y, z4));
            // get dynamic obsts for validation
            if (dynamic_obst_eval_env) {
                dynamic_obsts_start_poses.Add(new Vector3(x3, y, z1));
                dynamic_obsts_start_poses.Add(new Vector3(x1, y, z2));
                dynamic_obsts_start_poses.Add(new Vector3(x3, y, z3));
                dynamic_obsts_start_poses.Add(new Vector3(x1, y, z4));

                // dynamic_obsts_movings = List<Vector2>:  [( direction(1 forward, -1 backward), speed ), ...]
                // rb.MovePosition(transform.position + forward_vector * dirToGo * Time.deltaTime * new_movementSpeed);
                dynamic_obsts_movings.Add(new Vector2(-1, dynamicMovementSpeed));
                dynamic_obsts_movings.Add(new Vector2(1, dynamicMovementSpeed));
                dynamic_obsts_movings.Add(new Vector2(-1, dynamicMovementSpeed));
                dynamic_obsts_movings.Add(new Vector2(1, dynamicMovementSpeed));
            }


            // change scale of checkpoint
            checkpoint_box_col.size = new Vector3(50, 50, 50);
        }
        else if (env_type == 1) {
            if (agent_type == 1) {
                left_up_agent = new Vector3(138.7396f, 28.9f, -132.826f);
                right_down_agent = new Vector3(-133.2604f, 28.9f, 138.174f);
            }
            else if (agent_type == 2) {
                left_up_agent = new Vector3(130.7396f, 37.89999f, -129.826f);
                right_down_agent = new Vector3(-118.2604f, 37.89999f, 121.174f);
            }
        }
    }



    public override void CollectObservations(VectorSensor sensor) {
        if (env_type == 3) {
            Vector3 diff_checkpoint_agent = (checkpoint.localPosition - transform.localPosition);
            sensor.AddObservation(diff_checkpoint_agent);
        }
        else if (env_type == 1) {
            if (agent_type == 1) {
                Vector3 diff_snowman_agent = (snowman.localPosition - transform.localPosition);
                sensor.AddObservation(diff_snowman_agent);
            }
            else if (agent_type == 2) {
                Vector3 diff_agent_1_agent = (agent_1.localPosition - transform.localPosition);
                sensor.AddObservation(diff_agent_1_agent);
            }
        }
        sensor.AddObservation(transform.forward);
    }

    public override void OnEpisodeBegin() {

        // set agent start pose & checkpoint pose
        if (env_type == 1) {
            transform.localPosition = SampleRandomVector(left_up_agent, right_down_agent);
        }
        else if (env_type == 2) {
            checkPointMeetCounts = 0;
            transform.localPosition = new Vector3(-3.1f, 28.9f, -2.4f);
            checkpoint.transform.localPosition = new Vector3(405.2f, 28f, -17.7f);
        }
        else if (env_type == 3) {
            if (continue_train_task) {
                var inc_dec_array = new int[] {-1, 1};
                var inc_dec_random = inc_dec_array[Random.Range(0, inc_dec_array.Length)];
                if (inc_dec_random == 1) {
                    eval_env_while_train = true;
                }
                else {
                    eval_env_while_train = false;
                }
            }
            if (dynamic_obst_eval_env) {
                for (int i = 0; i < dynamic_obsts_start_poses.Count; i++) {
                    GameObject dynamic_obst = Instantiate(dynamicObst, dynamic_obsts_start_poses[i], Quaternion.identity, transform.parent);
                    dynamic_obst.transform.localPosition = dynamic_obsts_start_poses[i];
                    dynamic_obst.transform.rotation = Quaternion.LookRotation(new Vector3(1, 0, 0));
                    episodeDynamicObstacles.Add(dynamic_obst);
                }
            }
            if (eval_env || eval_env_while_train) {
                if (continue_train_task) {
                    var inc_dec_array = new int[] {-1, 1};
                    var inc_dec_random = inc_dec_array[Random.Range(0, inc_dec_array.Length)];
                    if (inc_dec_random == 1) {
                        reverse_eval_direction = true;
                    }
                    else {
                        reverse_eval_direction = false;
                    }
                }                
                maxSteps = 10000;
                checkPointMeetCounts = 0;
                if (reverse_eval_direction) {
                    transform.localPosition = goal_eval_pose;
                    checkpoint.localPosition = checkpoints[checkpoints.Count - checkPointMeetCounts - 1];
                }
                else {
                    transform.localPosition = start_eval_pose;
                    checkpoint.localPosition = checkpoints[checkPointMeetCounts];
                }
            }
            else {
                task_id = Random.Range(0, starts.Count);
                transform.localPosition = AddPertrubXZ(starts[task_id]);
                checkpoint.transform.localPosition = AddPertrubXZ(goals[task_id]);
                float rotationPetrub = Random.Range(0f, 360f);
                transform.Rotate(0f, rotationPetrub, 0f, Space.Self);
            }
        }
        stepCount = 0;

    }

    public void MoveAgent(ActionBuffers actions) {
        if (!descrete_actions) {
            float moveRotate = actions.ContinuousActions[0];
            float moveForward = actions.ContinuousActions[1];

            rb.MovePosition(transform.position + transform.forward * moveForward * Time.deltaTime * movementSpeed);
            transform.Rotate(0f, moveRotate * rotateSpeed * Time.deltaTime, 0f, Space.Self);
        }
        else {
            float dirToGo = 0;
            float rotateDir = 0;
            var dirToGoForwardAction = actions.DiscreteActions[0];
            var rotateDirAction = actions.DiscreteActions[1];
            float new_movementSpeed = movementSpeed;
            float new_rotateSpeed = rotateSpeed;
            Vector3 forward_vector = transform.forward;
            // move forward
            if (dirToGoForwardAction == 1) {
                dirToGo = 1f;
            }
            else if (dirToGoForwardAction == 2) {
                new_movementSpeed = 0;
            }
            else {
                // dirToGoForwardAction == 3
                dirToGo = -1f;
            }

            // rotate
            if (rotateDirAction == 1) {
                rotateDir = 1f;
            }
            else if (rotateDirAction == 2) {
                new_rotateSpeed = 0;
            }
            else {
                // dirToGoForwardAction == 3
                rotateDir = -1f;
                
            }
            
            // slide
            if (add_slide) {
                var dirToGoSideAction = actions.DiscreteActions[2];
                if (dirToGoSideAction == 1) {
                    forward_vector = forward_vector - transform.right;
                }
                else if (dirToGoSideAction == 2) {
                }
                else {
                    forward_vector = forward_vector + transform.right;
                }
                
            }

            rb.MovePosition(transform.position + forward_vector * dirToGo * Time.deltaTime * new_movementSpeed);
            transform.Rotate(0f, rotateDir * new_rotateSpeed * Time.deltaTime, 0f, Space.Self);
            //rb.AddForce(dirToGo * new_movementSpeed, ForceMode.VelocityChange);

        }
        
    }

    public void MoveDynamics() {
        GameObject dynamic_obst;
        Rigidbody rb_dynamic;
        for (int i = 0; i < episodeDynamicObstacles.Count; i++) {
            dynamic_obst = episodeDynamicObstacles[i];
            rb_dynamic = dynamic_obst.GetComponent<Rigidbody>();
            Vector3 dynamic_forward_vector = dynamic_obst.transform.forward;
            float dynamic_dirToGo = dynamic_obsts_movings[i].x;
            float dynamic_movementSpeed = dynamic_obsts_movings[i].y;
            rb_dynamic.MovePosition(dynamic_obst.transform.position + dynamic_forward_vector * dynamic_dirToGo * Time.deltaTime * dynamic_movementSpeed);
        }
    }

    public override void OnActionReceived(ActionBuffers actions) {

        // debug
        //Debug.Log("dyn obst count: " + episodeDynamicObstacles.Count);
        /*
        for (int i = 0; i < episodeDynamicObstacles.Count; i++) {
            Debug.Log("dyn obs: " + i + " x: " + episodeDynamicObstacles[i].transform.forward.x + 
                            " y: " + episodeDynamicObstacles[i].transform.forward.y + " z: " + episodeDynamicObstacles[i].transform.forward.z);
        }
        Debug.Log("agent: " + " x: " + transform.forward.x + " y: " + transform.forward.y  + " z: " + transform.forward.z);
        */ 
        
        Vector3 prev_pose = rb.position;

        // action = rotate, move_forward
        MoveAgent(actions);

        // move dynamic obsts
        if (dynamic_obst_eval_env) {
            MoveDynamics();
        }

        Vector3 current_pose = rb.position;
        
        float reward = 0;
        // Euclid Reward
        if (env_type == 3) {
            // checkpoint reward
            float prevDistanceToTarget = Vector3.Distance(prev_pose, checkpoint.localPosition);
            float currentDistanceToTarget = Vector3.Distance(current_pose, checkpoint.localPosition);
            reward += prevDistanceToTarget - currentDistanceToTarget;    
        }
        else if (env_type == 1) {
            if (agent_type == 1) {
                // run from snowman
                float prevDistanceToSnowman = Vector3.Distance(prev_pose, snowman.localPosition);
                float currentDistanceToSnowman = Vector3.Distance(current_pose, snowman.localPosition);
                reward += currentDistanceToSnowman - prevDistanceToSnowman; 
            }
            else if (agent_type == 2) {
                // catch the agent
                float prevDistanceToAgent = Vector3.Distance(prev_pose, agent_1.localPosition);
                float currentDistanceToAgent = Vector3.Distance(current_pose, agent_1.localPosition);
                reward += prevDistanceToAgent - currentDistanceToAgent;  
            }
        }
        
        // Action Reward
        if (descrete_actions) {
            
        }
        else {
            float moveRotate = actions.ContinuousActions[0];
            float moveForward = actions.ContinuousActions[1];
            reward -= System.Math.Abs(moveRotate) / 3;
            reward -= System.Math.Abs(moveForward) / 3;
        }
        // Timestep reward
        reward = reward - 0.1f;

        SetReward(reward);
        // test
        //Debug.Log("diff x: " + (transform.localPosition.x - prev_pose.x));
        //Debug.Log("reward: " + reward);
        //Debug.Log("forward x: " + transform.forward.x + " y: " + transform.forward.y + " z: " + transform.forward.z);

        stepCount++;
        if (stepCount >= maxSteps)
        {
            var statsRecorder = Academy.Instance.StatsRecorder;
            statsRecorder.Add("collision", 0.0f);
            OnEpisodeEnd();
            EndEpisode();
        }

    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        if (!descrete_actions) {
            ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");
            continuousActions[1] = Input.GetAxis("Vertical");
        }
        else {
            ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

            float horizontalInput = Input.GetAxis("Vertical");
            int horizontalInt;
            if (horizontalInput > 0.1f) {
                horizontalInt = 1;
            } else if (horizontalInput < -0.1f) {
                horizontalInt = 3;
            } else {
                horizontalInt = 2;
            }

            float verticalInput = Input.GetAxis("Horizontal");
            int verticalInt;
            if (verticalInput > 0.1f) {
                verticalInt = 1;
            } else if (verticalInput < -0.1f) {
                verticalInt = 3;
            } else {
                verticalInt = 2;
            }

            if (add_slide) {
                if (Input.GetButton("Fire1")) {
                    discreteActions[2] = 1;
                }
                else if (Input.GetButton("Fire2")) {
                    discreteActions[2] = 3;
                }
                else {
                    discreteActions[2] = 2;
                }
            }


            discreteActions[0] = horizontalInt;
            discreteActions[1] = verticalInt;
        }
    }

    public void OnEpisodeEnd() {
        if (dynamic_obst_eval_env) {
            for (int i = 0; i < episodeDynamicObstacles.Count; i++) {
                Destroy(episodeDynamicObstacles[i]);
            }
            episodeDynamicObstacles.Clear();
        }
    }


    private void OnTriggerExit(Collider other) {
        if (other.TryGetComponent(out CheckPoint _)) {
            isTriggeredCheckPoint = false;
        }
        if (other.TryGetComponent(out AgentToTarget AgentToTarget)) {
            isTriggeredOtherAgent = false;
        }
    }

    private void OnTriggerEnter(Collider other) {
        if (other.TryGetComponent(out Goal Goal)) {
            AddReward(50f);
            OnEpisodeEnd();
            EndEpisode();
        }
        else if (other.TryGetComponent(out Wall Wall) || other.TryGetComponent(out DynamicObstBehaviour _)) {
            AddReward(-100f);
            var statsRecorder = Academy.Instance.StatsRecorder;
            statsRecorder.Add("collision", 1.0f);
            OnEpisodeEnd();
            EndEpisode();
        }
        else if (other.TryGetComponent(out CheckPoint _) && !isTriggeredCheckPoint) {
            isTriggeredCheckPoint = true;
            AddReward(50f);
            if (env_type == 3) {
                if (eval_env || eval_env_while_train) {
                    checkPointMeetCounts += 1;
                    if (checkPointMeetCounts >= checkpoints.Count) {
                        OnEpisodeEnd();
                        EndEpisode();
                    }
                    else {
                        if (reverse_eval_direction) {
                            checkpoint.localPosition = checkpoints[checkpoints.Count - checkPointMeetCounts - 1];    
                        }
                        else {
                            checkpoint.localPosition = checkpoints[checkPointMeetCounts];
                        }
                    }
                }
                else {
                    var statsRecorder = Academy.Instance.StatsRecorder;
                    statsRecorder.Add("collision", 0.0f);
                    OnEpisodeEnd();
                    EndEpisode();
                }
            }
            else if (env_type == 2) {
                checkPointMeetCounts += 1;
                if (checkPointMeetCounts == 1) {
                    checkpoint.transform.localPosition = new Vector3(592f, 28.2f, -405f);
                }
                else if (checkPointMeetCounts == 2) {
                    checkpoint.transform.localPosition = new Vector3(796f, 28.2f, -71f);
                }
                else if (checkPointMeetCounts == 3) {
                    checkpoint.transform.localPosition = new Vector3(1194f, 36.2f, 496f);
                }
            }
        }
        if (other.TryGetComponent(out AgentToTarget AgentToTarget)) {
            if (env_type == 1) {
                if (agent_type == 1 && AgentToTarget.agent_type == 2) {
                    AddReward(-100f);
                    Debug.Log("add agent penalty");
                    //OnEpisodeEnd();
                    //EndEpisode();
                }
                else if (agent_type == 2 && AgentToTarget.agent_type == 1) {
                    AddReward(50f);
                    Debug.Log("add snowman reward");
                    //OnEpisodeEnd();
                    //EndEpisode();
                }
            }
        
        }
    }
}
