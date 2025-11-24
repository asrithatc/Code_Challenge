
---

# 📘 (Optional) **answers/extra_credit_aws.md**

```markdown
# Extra Credit — AWS Deployment Plan

## API Deployment
- Containerize FastAPI using Docker
- Deploy on **AWS ECS Fargate**
- Use **Application Load Balancer** for routing
- Manage logs through CloudWatch

## Database
Use **Amazon RDS PostgreSQL**, Multi-AZ enabled.

## Ingestion
Scheduled ingestion via:
- **Amazon ECS Scheduled Tasks** (EventBridge)
- Task pulls raw files from **S3**
- Writes into RDS

## Infrastructure as Code
Use **Terraform** or **AWS CDK** to:
- Create ECS cluster
- Configure networking (VPC/Subnets/Security Groups)
- Deploy RDS
- Deploy the service

## Summary
This architecture is fully serverless-compatible, scalable, and production-ready.
