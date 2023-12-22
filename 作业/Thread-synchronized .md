五个人拿着钱去买票，五块钱一张票，五个人只有面值为5，10，20的钞票。当顾客买票时售票员能够进行找零，就把票卖出去，否则就让此顾客等待，qu服务其他顾客，等收了其他乘客的钞票后再看看能不能对等待的顾客进行找零。

```java
//售票员
class Salesman {
    private String name;

    //只有一张五块的
    private int money5 = 1, money10 = 0, money20 = 0;
    private int balance;

    public int getMoney5() {
        return money5;
    }
    //收一张五块的
    public void setMoney5(int money5) {
        this.money5 = money5;
    }

    public int getMoney10() {
        return money10;
    }

    //收一张十块的
    public void setMoney10(int money10) {
        this.money10 = money10;
    }

    public int getMoney20() {
        return money20;
    }

    //收一张10块的
    public void setMoney20(int money20) {
        this.money20 = money20;
    }

    public int getBalance() {
        this.setBalance();
        return balance;
    }

    //售票员总资金
    public String myMoney() {
        return "*****我现在有 5元: " + this.getMoney5() +
                ", 我现在有 10元: " + this.getMoney10() +
                ", 我现在有 20元: " + this.getMoney20() +
                ", 总计 " + this.getBalance();

    }

    public void setBalance() {
        this.balance = money5 * 5 + money10 * 10 + money20 * 20;
    }

    public Salesman(String name, int money5, int money10, int money20) {
        this.name = name;
        this.money5 = money5;
        this.money10 = money10;
        this.money20 = money20;
        this.setBalance();
    }

    /**
     *
     * @param recvMoney
     * @param ticketNum
     */
    public synchronized void sell(int recvMoney, int ticketNum) {
        if (recvMoney == 5 && ticketNum == 1) { //直接卖票
            System.out.println(Thread.currentThread().getName() +
                    "拿着5块钱来买1张票");
            this.setMoney5(this.getMoney5() + 1);
            System.out.println(Thread.currentThread().getName() +"\t不用找零, 您好这是你的 " + ticketNum + " 张票");
            System.out.println(myMoney());
        } else if (recvMoney == 10 && ticketNum == 2) { //直接卖票
            System.out.println(Thread.currentThread().getName() +
                    "拿着10块钱来买2张票");
            this.setMoney10(this.getMoney10() + 1);
            System.out.println(Thread.currentThread().getName() +"\t不用找零, 您好这是你的 " + ticketNum + " 张票");
            System.out.println(myMoney());
        } else if (recvMoney == 10 && ticketNum == 1) {
            while (this.getMoney5() < 1) {
                try {
                    System.out.println(Thread.currentThread().getName() +
                            "拿着10块钱来买1张票");
                    System.out.println(myMoney() + " \n\t\t" +
                            Thread.currentThread().getName() + " 没零钱，等着叫你找钱");
                    //没有零钱找，就进入等待
                    wait();
                    System.out.println("\t\t" + Thread.currentThread().getName() +
                            "__结束等待");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println(Thread.currentThread().getName() +
                    "拿着10块钱来买1张票");
            this.setMoney5(this.getMoney5() - 1);
            this.setMoney10(this.getMoney10() + 1);
            System.out.println(Thread.currentThread().getName() +"\t 给你找零5块, 您好这是你的 " + ticketNum + " 张票");
            System.out.println(myMoney());
        } else if (recvMoney == 20 && ticketNum == 2) {

            while (this.getMoney10() < 1) {
                try {
                    System.out.println(Thread.currentThread().getName() +
                            "拿着20块钱来买2张票");
                    System.out.println(myMoney() + " \n\t\t" +
                            Thread.currentThread().getName() + " 没零钱，等着叫你找钱");
                    //没钱找，进入等待
                    wait();
                    System.out.println("\t\t" + Thread.currentThread().getName() +
                            "__结束等待");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            this.setMoney10(this.getMoney10() - 1);
            this.setMoney20(this.getMoney20() + 1);
            System.out.println(Thread.currentThread().getName() +"\t给你找零10块, 您好这是你的 " + ticketNum + " 张票");
            System.out.println(myMoney());
        } else if (recvMoney == 20 && ticketNum == 1) {
            //该线程就会沿着wait方法之后的路径继续执行

            while (!(this.getMoney10() >= 1 && this.getMoney5() >= 1)) {
                try {
                    System.out.println(Thread.currentThread().getName() +
                            "拿着20块钱来买1张票");
                    System.out.println(myMoney() + " \n\t\t" +
                            Thread.currentThread().getName() + " 没零钱，等着叫你找钱");
                    //没零钱找，进入等待
                    wait();
                    System.out.println("\t\t" + Thread.currentThread().getName() +
                            "__结束等待");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            this.setMoney5(this.getMoney5() - 1);
            this.setMoney10(this.getMoney10() - 1);
            this.setMoney20(this.getMoney20() + 1);
            System.out.println(Thread.currentThread().getName() +"\t给你找零15块, 您好这是你的 " + ticketNum + " 张票");
            System.out.println(myMoney());
        }
        //唤醒所有等待的进程
        notifyAll();
    }
}
//顾客类
class Customer extends Thread {

    private String Cname;
    private int tickets; //当前customer需要买的票的张数
    private int money5, money10, money20;
    private int balance;
    private Salesman salesman;

    public int getBalance() {
        this.setBalance();
        return balance;
    }

    public String getCname() {
        return Cname;
    }

    public void setCame(String name) {
        this.Cname = name;
    }

    public void setBalance() {
        this.balance = money5 * 5 + money10 * 10 + money20 * 20;
    }

    public int getTickets() {
        return tickets;
    }

    public void setTickets(int tickets) {
        this.tickets = tickets;
    }

    public int getMoney5() {
        return money5;
    }

    public void setMoney5(int money5) {
        this.money5 = money5;
    }

    public int getMoney10() {
        return money10;
    }

    public void setMoney10(int money10) {
        this.money10 = money10;
    }

    public int getMoney20() {
        return money20;
    }

    public void setMoney20(int money20) {
        this.money20 = money20;
    }

    public Customer(String name, int tickets, int money5, int money10, int money20, Salesman salesman) {
        this.Cname = name;
        this.tickets = tickets;
        this.money5 = money5;
        this.money10 = money10;
        this.money20 = money20;
        this.salesman = salesman;
    }

    @Override
    public void run() {
        //拿着钱去找售票员卖票
        salesman.sell(this.getBalance(), this.getTickets());
    }
}

class Sale {
    public static void main(String[] args) {
        Salesman salesman = new Salesman("售票员", 1, 0, 0);

        Customer Zhao = new Customer("赵", 2, 0, 0, 1, salesman);
        Zhao.setName("赵");
        Customer Qian = new Customer("钱", 1, 0, 0, 1, salesman);
        Qian.setName("钱");
        Customer Sun = new Customer("孙", 1, 0, 1, 0, salesman);
        Sun.setName("孙");
        Customer Li = new Customer("李", 2, 0, 1, 0, salesman);
        Li.setName("李");
        Customer Zhou = new Customer("周", 1, 1, 0, 0, salesman);
        Zhou.setName("周");

        Zhao.start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        Qian.start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        Sun.start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        Li.start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        Zhou.start();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
    }
}
```

结果：

![](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023185452207.png)

现在对结果进行分析：

赵拿着20块买2张票，找不开，先等着

钱拿着20块买1张票，找不开，先等着

孙拿着10块买一张票，能找开，买完票，然后唤醒等待的进程（赵和钱）

钱还是找不开，接着等待

但此时赵能找开了，买完票后唤醒等待进程（钱）

钱还是找不开，进入等待

李拿着10块买2张票，能找开，买完票，唤醒等待进程（钱）

此时钱还是找不开，接着等待

周拿着5块买张票，能找开，买完票，唤醒等待进程（钱）

此时钱终于能找开了，买完票

整个程序结束。

每个进程启动的时候之间会有一段sleep()，用来表示排队买票。