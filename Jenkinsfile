pipeline {
  agent {
    docker {
      image 'hls4ml-with-vivado:latest'
    }
  }
  options {
    timeout(time: 6, unit: 'HOURS')
  }

  stages {
    stage('Keras to HLS') {
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash
              . activate hls4ml-py36
              pip install -U ../ --user
              ./convert-keras-models.sh -p 3 -x -f keras-models.txt
              pip uninstall hls4ml -y'''
        }
      }
    }
    stage('C Simulation') {
      parallel {
        stage('2017.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2017', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2017.2 -c -p 2'''
            }
          }
        }
        stage('2018.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2018', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2018.2 -c -p 2'''
            }
          }
        }
      }
    }
    stage('C/RTL Synthesis') {
      parallel {
        stage('2017.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2017', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2017.2 -s -r -p 2'''
            }
          }
        }
        stage('2018.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2018', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2018.2 -s -r -p 2'''
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
