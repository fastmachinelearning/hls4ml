pipeline {
  agent {
    docker {
      image 'vivado-alma9:1'
      args  '-v /data/Xilinx:/data/Xilinx'
    }
  }
  options {
    timeout(time: 6, unit: 'HOURS')
  }

  stages {
    stage('Keras to HLS') {
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash --login
              conda activate hls4ml-py310
              conda install -y jupyterhub pydot graphviz pytest pytest-cov
              pip install pytest-randomly jupyter onnx>=1.4.0 matplotlib pandas seaborn pydigitalwavetools==1.1 pyyaml tensorflow==2.14 qonnx torch git+https://github.com/jmitrevs/qkeras.git@qrecurrent_unstack pyparsing
              pip install -U ../ --user
              ./convert-keras-models.sh -x -f keras-models.txt
              pip uninstall hls4ml -y'''
        }
      }
    }
    stage('C Simulation') {
      parallel {
        stage('2019.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2019', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2019.2 -c -p 2'''
            }
          }
        }
        stage('2020.1') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2020', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2020.1 -c -p 2'''
            }
          }
        }
      }
    }
    stage('C/RTL Synthesis') {
      parallel {
        stage('2019.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2019', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2019.2 -s -r -p 2'''
            }
          }
        }
        stage('2020.1') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2020', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2020.1 -s -r -p 2'''
            }
          }
        }
      }
    }
    stage('Report') {
      when {
        environment name: 'TEST_SYNTHESIS', value: '1'
      }
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash
             ./gather-reports.sh -b | tee report.rpt'''
        }
        archiveArtifacts artifacts: 'test/report.rpt', fingerprint: true
      }
    }
  }
  post {
    always {
      dir(path: 'test') {
          sh '''#!/bin/bash
             ./cleanup.sh'''
      }
    }
  }
}
